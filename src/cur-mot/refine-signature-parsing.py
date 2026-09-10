#!/usr/bin/env python3
## nb not run in the normal swerik env
## Spacy numpy dependency not compatible with our tensorflow/numpy version
"""
Use heuristics and NER to parse and classify signature elements
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # hides all GPUs
os.environ["SPACY_FORCE_CPU"] = "true"      # newer spaCy obeys this
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["THINC_NO_OPTIMIZE"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import multiprocessing as mp

# Must come before any Pool, Executor, or spaCy import
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)
from multiprocessing import (
    get_context,
    Process,
)
from pyriksdagen.args import (
    fetch_parser,
    impute_args
)
from pyriksdagen.io import (
    parse_tei,
    write_tei,
)
from tqdm import tqdm
from typing import (
    List,
    Optional,
    Tuple,
)
import sys, time
#import cupy as cp
import gc
import lxml.etree as etree
import re
import spacy
import thinc
from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans
from spacy.tokens import Span
from spacy.language import Language
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
import traceback
UP = "A-ZÅÄÖÉÈÀÂÊÎÔÛÜÖÄÅ"
LO = "a-zåäöéèàâêîôûüöäåç"

NAME_TOKEN = re.compile(rf"^[{UP}][{LO}{UP}\-'\.:]+$")
SHORT_PAREN = re.compile(r"^\([A-Za-zÅÄÖåäö]{1,3}\)$")
RE_NR_HEAD = re.compile(r"^(?:N:o|Nr)$", re.IGNORECASE)
RE_ALNUM   = re.compile(r"^(?:[A-Za-zÅÄÖåäö]+|\d+)$")

# Allow longer multi-word places after 'i' / 'från'
MAX_PLACE_TOKENS = 6
LOWER_PLACE_WORDS = {
    "län","härad","socken","församling","kommun","stad","köping",
    "tingslag","domsaga","landskap","kapellag","bygden"
}
# Optional surname hints (we still allow 2-token names even if 2nd isn’t in this list)
SURNAME_SUFFIXES = (
    "son","sson","dotter","berg","borg","gren","kvist","quist",
    "ström","man","mark","lund","holm","beck","blad","feldt","felt",
    "fors","vall","hage","dahl"
)
NOBILIARY_PARTICLES = {"von", "de", "af", "van", "di", "du", "v"}
PLACE_HINT = re.compile(r"^[A-ZÅÄÖ][a-zåäöéèàâêîôûüöäåç\-]+$")
PLACE_ENDINGS = {"län", "stad", "kommun", "församling", "härad", "socken", "kapellag"}

def log(msg):
    sys.stdout.write(f"[PID {os.getpid()}] {msg}\n")
    sys.stdout.flush()


def clean_text(t: str) -> str:
    # Fix broken hyphenation (Carls- son → Carlsson); normalize whitespace/dashes.
    t = re.sub(rf"([{UP}{LO}])-\s+([{UP}{LO}])", r"\1\2", t)
    t = t.replace("’","'").replace("–","-").replace("—","-")
    t = re.sub(r"\b0\.", "O.", t)
    return re.sub(r"\s+"," ", t.strip())


def is_initial(tok: str) -> bool:
    t = tok.strip()
    return bool(re.fullmatch(rf"[{UP}]\.?", t))


def is_hy_initial(tok: str) -> bool:
    t = tok.strip()
    return bool(re.fullmatch(rf"-[{UP}]\.?", t))


def is_name_token(tok: str) -> bool:
    return bool(NAME_TOKEN.match(tok))


def is_surname_like(tok: str) -> bool:
    t = tok.lower().rstrip(".")
    return any(t.endswith(s) for s in SURNAME_SUFFIXES)


# --- spaCy helper (optional) -------------------------------------------------
def _spacy_doc(text: str):
    try:
        import spacy
        for m in ("sv_core_news_md",):
            try:
                return spacy.load(m)(text)
            except Exception:
                pass
        return None
    except Exception:
        return None


def _ent_covering(doc, start_char: int) -> Optional[Tuple[str,int,int]]:
    """Return (label, ent_start_char, ent_end_char) for entity starting at or covering start_char."""
    if not doc: return None
    for e in doc.ents:
        if e.start_char <= start_char < e.end_char and e.label_ in ("PER","LOC","GPE"):
            return (e.label_, e.start_char, e.end_char)
    return None


# --- tokenization with spans -------------------------------------------------
def _tokenize_with_spans(text: str):
    toks = text.split()
    spans = []
    pos = 0
    for tok in toks:
        start = pos
        end = start + len(tok)
        spans.append((tok, start, end))
        pos = end + 1  # +1 for the single space between tokens
    return toks, spans


def _consume_record_id(toks, i):
    # Expect head: "Nr" or "N:o"
    if i >= len(toks) or not RE_NR_HEAD.match(toks[i]):
        return None
    parts = [toks[i]]; j = i + 1
    # up to two following parts that are letters or digits (to catch "B 236")
    take = 0
    while j < len(toks) and take < 2 and RE_ALNUM.match(toks[j]):
        parts.append(toks[j])
        j += 1; take += 1
    if take == 0:
        return None
    return j, " ".join(parts)


# --- consume place phrase after 'i' / 'från' --------------------------------
def _consume_place(toks, i_start,use_spacy_ent, spans) -> int:
    """
    Consume tokens after 'i' or 'från' that look like a place.
    Stops before what appears to be the next person's name.
    """
    i = i_start + 1
    if i >= len(toks):
        return i

    # If spaCy already detected a location entity covering the next token
    if use_spacy_ent:
        label, ent_s, ent_e = use_spacy_ent
        if label in ("LOC", "GPE"):
            j = i
            while j < len(toks) and spans[j][2] <= ent_e:
                j += 1
            return max(j, i)

    taken = 0
    j = i
    while j < len(toks):
        w = toks[j]

        # Allow lowercase place-type words (län, härad, socken, etc.)
        if w.lower() in LOWER_PLACE_WORDS:
            j += 1
            taken += 1
            continue

        # Stop if we hit "Nr" or record patterns
        if RE_NR_HEAD.match(w) or w.lower().startswith("nr"):
            break

        # Allow capitalized place components (Stockholms, Västra, Sundbyberg)
        if re.match(r"^[A-ZÅÄÖ][a-zåäöéèàâêîôûüöäåç\-]*$", w):
            # stop early if this looks like a person surname (e.g. ends in -son, -berg)
            if is_surname_like(w):
                break
            j += 1
            taken += 1
            continue

        break

    # Must have taken at least one token to count as a place
    if taken == 0:
        return i_start + 1

    return j


@Language.factory("known_name_ruler")
def create_known_name_ruler(nlp, name, known_names=None, known_places=None):
    """Factory for a component that adds known person/place entities."""
    person_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    if known_names:
        person_matcher.add("PER", [nlp.make_doc(n) for n in known_names])

    place_matcher = None
    if known_places:
        place_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        place_matcher.add("LOC", [nlp.make_doc(p) for p in known_places])

    # The actual callable component
    def known_name_component(doc):
        new_ents = []

        for _, start, end in person_matcher(doc):
            new_ents.append(Span(doc, start, end, label="PER"))

        if place_matcher:
            for _, start, end in place_matcher(doc):
                new_ents.append(Span(doc, start, end, label="LOC"))

        doc.ents = filter_spans(list(doc.ents) + new_ents)
        return doc

    return known_name_component


def add_known_names_ruler(nlp, known_names, known_places=None):
    """Attach known-name/place ruler safely."""
    if "known_name_ruler" not in nlp.pipe_names:
        nlp.add_pipe(
            "known_name_ruler",
            before="ner",
            config={"known_names": known_names, "known_places": known_places},
        )
    return nlp


def is_name_boundary(tok: str) -> bool:
    """
    Decide if token likely begins a new signature block or record section.
    """
    # new signature usually starts after a name (capitalized) *and* previous token ended cleanly
    return bool(
        tok.lower() in {"i", "nr", "från"} or
        tok in {",", ";"} or
        re.match(r"^[-–]$", tok) or
        is_initial(tok)
    )


def _merge_adjacent(items):
    merged = []
    for it in items:
        if merged and merged[-1]["type"] == it["type"]:
            # avoid duplicate words
            prev = merged[-1]["text"].split()
            new = it["text"].split()
            if not new or new[0] in prev:
                continue
            merged[-1]["text"] = " ".join(prev + new)
        else:
            merged.append(it)
    return merged


def merge_overlapping_spans(spans):
    """Merge overlapping or nested spaCy spans."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x.start_char, x.end_char))
    merged = [spans[0]]
    for cur in spans[1:]:
        prev = merged[-1]
        if cur.start_char <= prev.end_char:
            merged[-1] = doc.char_span(prev.start_char, max(prev.end_char, cur.end_char))
        else:
            merged.append(cur)
    return merged


# --- main ordered parser -----------------------------------------------------
def parse_name_string(text: str, nlp=None, doc=None, known_names=None) -> List[dict]:
    text = clean_text(text)
    if not text:
        return []

    doc = nlp(text)
    toks, spans = _tokenize_with_spans(text)
    # Ensure every span is a 3-tuple (tok, start, end)
    fixed_spans = []
    for s in spans:
        if isinstance(s, (list, tuple)) and len(s) == 3:
            fixed_spans.append(tuple(s))
        else:
            # fall back to dummy start/end positions
            if isinstance(s, str):
                fixed_spans.append((s, 0, len(s)))
            else:
                fixed_spans.append(("", 0, 0))
    spans = fixed_spans
    spans = [t if len(t) == 3 else (t[0], 0, 0) for t in spans]
    out = []
    i = 0

    # # build matcher if we have a list of names
    # name_spans = []
    # if known_names:
    #     matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    #     patterns = [nlp.make_doc(name.lower()) for name in known_names if isinstance(name, str)]
    #     matcher.add("KNOWN_NAME", patterns)
    #     for _, start, end in matcher(doc):
    #         name_spans.append(doc[start:end])

    # merge spaCy PER entities with known-name matches
    name_spans = [ent for ent in doc.ents if ent.label_ == "PER"]
    name_spans = sorted(name_spans, key=lambda e: e.start_char)
    loc_spans  = [ent for ent in doc.ents if ent.label_ in ("LOC", "GPE")]
    # index entities by start offset for quick lookup
    ent_starts = {e.start_char: e for e in name_spans}
    loc_starts = {e.start_char: e for e in loc_spans}

    while i < len(toks):
        try:
            tok, s, e = spans[i]
            if not isinstance(tok, str):
                tok = str(tok)
            if not isinstance(s, int) or not isinstance(e, int):
                s, e = 0, len(tok)
        except Exception:
            # fallback if the tuple isn't valid
            val = spans[i]
            if isinstance(val, str):
                tok, s, e = val, 0, len(val)
            elif isinstance(val, (list, tuple)):
                tok = str(val[0]) if len(val) > 0 else ""
                s = int(val[1]) if len(val) > 1 and isinstance(val[1], int) else 0
                e = int(val[2]) if len(val) > 2 and isinstance(val[2], int) else s + len(tok)
            else:
                tok, s, e = "", 0, 0

        # ---- record id -------------------------------------------------------
        two = " ".join(toks[i:i+2])
        if re.match(r"(?i)\bN[:r]\s*\d+\b", two):
            out.append({"type": "record-id", "text": two})
            i += 2
            continue

        # ---- location specifier ---------------------------------------------
        if tok.lower() in {"i", "från"}:
            use_ent = None
            for ent in loc_spans:
                if ent.start_char <= spans[i][1] < ent.end_char:
                    use_ent = (ent.label_, ent.start_char, ent.end_char)
                    break

            j = _consume_place(toks, i, use_ent, spans)
            frag = " ".join(toks[i:j])
            out.append({"type": "location-specifier", "text": frag})
            i = j
            continue

        # ---- matched entity -------------------------------------------------
        if s in ent_starts:
            ent = ent_starts[s]
            out.append({"type": "person", "text": ent.text})
            # advance i safely past the entity span
            j = i
            while j < len(spans) and spans[j][2] <= ent.end_char:
                j += 1
            i = j
            continue

 # ---- initials-aware name grouping with strong sequential rules -------
        if re.match(r"^[A-ZÅÄÖ]", tok):
            j = i
            name_tokens = []

            def is_initial(t):
                """A single capital, with or without dot, e.g. 'W' or 'W.'"""
                return bool(re.fullmatch(r"[A-ZÅÄÖ]\.?", t))

            def is_word(t):
                """A proper name-like token."""
                return bool(re.match(r"^[A-ZÅÄÖ][a-zåäö\-]+$", t))

            while j < len(toks):
                nxt = toks[j]

                # Hard stop markers
                if nxt.lower() in {"i", "från"} or nxt in {"Nr", "nr"}:
                    break
                if re.fullmatch(r"[,;]", nxt):
                    j += 1
                    continue

                # Accept initials or capitalized words
                if is_initial(nxt) or is_word(nxt) or re.fullmatch(r"[A-ZÅÄÖ][a-zåäö]{1,3}\.", nxt):
                    name_tokens.append(nxt)
                    j += 1

                    # --- Rule 1: "initials + word" followed by another initial → new name
                    if (
                        len(name_tokens) >= 2
                        and is_initial(name_tokens[-2])
                        and is_word(name_tokens[-1])
                        and j < len(toks)
                        and is_initial(toks[j])
                    ):
                        break

                    # --- Rule 2: prevent ending on an initial (with or without dot)
                    # keep going until we get a surname-like word
                    if j < len(toks) and is_initial(name_tokens[-1]) and not is_word(toks[j]):
                        continue

                    # --- Rule 3: detect next full-name start ("Firstname" pattern)
                    if (
                        j < len(toks)
                        and is_word(name_tokens[-1])
                        and is_word(toks[j])
                        and j + 1 < len(toks)
                        and (is_initial(toks[j + 1]) or is_word(toks[j + 1]))
                    ):
                        break

                    continue

                break  # anything else stops

            frag = " ".join(name_tokens).strip()
            if frag:
                out.append({"type": "person", "text": frag})
            i = j
            continue

        i += 1

    del doc
    return out



def looks_like_signature_block(sig_block, nlp=None):
    texts = " ".join([itm.strip() for itm in sig_block.itertext() if itm is not None and itm.strip()!=''])
    #print(texts)
    if not texts:
        #print("no texts")
        return False

    if not any([c.isupper() for c in texts]):
        #print("no uppercase")
        return False

    words = texts.split()
    #print(words)
    if len(words) < 4:
        # trivial, short, might be single name
        return True

    # reject only if this looks like pure prose
    avg_len = sum(len(w) for w in words) / len(words)
    caps_ratio = sum(w and w[0].isupper() for w in words) / len(words)

    # allow if many short capitalized tokens
    if caps_ratio > 0.5:# and avg_len < 10:
        return True

    if len(words) > 200:
        #print("too many words")
        #print(texts)
        return False   # extreme prose

    # Always accept anything that has even 1 typical surname or capitalized run
    if any(w.endswith(("sson","berg","man","gren","lund","ström")) for w in words):
        return True

    if nlp:
        doc = nlp(texts)
        if any(ent.label_ == "PER" for ent in doc.ents):
            return True

    return True   # << default to True to test the rest of the pipeline


def expand_signatures(sig_block, ns, nlp, parser_fn=parse_name_string, known_names=None):
    changed = False
    try:
        if not looks_like_signature_block(sig_block, nlp):
            return changed

        lists = sig_block.findall(f".//{ns['tei_ns']}list")
        for lst in lists:
            old_items = list(lst.findall(f"{ns['tei_ns']}item"))
            if not old_items:
                continue

            full_text = " ".join(
                (itm.text or "").strip()
                for itm in old_items
                if (itm.text or "").strip()
            ).strip()
            if not full_text:
                continue

            parsed = parser_fn(full_text, nlp)

            new_items = []
            for entry in parsed:
                el = etree.Element("item")
                el.text = entry["text"]
                t = entry["type"]
                if t == "person":
                    el.set("type", "signature")
                elif t in {"location", "location-specifier"}:
                    el.set("type", "location-specifier")
                elif t == "record_id":
                    el.set("type", "record-id")
                new_items.append(el)

            lst[:] = new_items
            changed = True

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[expand_signatures] ⚠ {e}\n{tb}")
        raise   # propagate so process_one can report it

    return changed

def _load_spacy():
    try:
        nlp = spacy.load("sv_core_news_md", disable=["parser","lemmatizer"])
    except OSError:
        raise Error("you want to run with spacy...install it :D")
    else:
        df0 = pd.read_csv("riksdagen-persons/data/name.csv")
        known_names = df0["name"].unique().tolist()
        df1 = pd.read_csv("riksdagen-persons/data/location_specifier.csv")
        known_places = df1["location"].unique().tolist()
        nlp = add_known_names_ruler(nlp, known_names, known_places=known_places)
        return nlp, known_names


def get_nlp():
    # Each process gets its own nlp instance, loaded once
    if not hasattr(get_nlp, "_nlp"):
        get_nlp._nlp, get_nlp._known_names = _load_spacy()  # returns (nlp, known_names)
    return get_nlp._nlp, get_nlp._known_names


def process_one(args):
    path, cfg = args
    try:
        nlp, known_names = _load_spacy()
    except Exception as e:
        return (path, False, f"nlp init failed: {e}")

    root = None
    try:
        root, ns = parse_tei(path)
        changed = expand_signatures(root, ns, nlp, known_names=known_names)
        if changed:
            write_tei(root, path)
        return (path, bool(changed), None)
    except Exception as e:
        tb = traceback.format_exc()
        return (path, False, f"{e}\n{tb}")
    finally:
        try:
            if root is not None:
                root.clear()
        except Exception:
            pass
        gc.collect()

_worker_nlp = None
_worker_known_names = None

def worker_init(known_names_csv, known_places_csv):
    """Runs once in each worker process."""
    # make workers single-threaded (prevents BLAS/OMP deadlocks)
    import os, sys
    msg = f"[PID {os.getpid()}] worker init starting\n"
    sys.stdout.write(msg)
    sys.stdout.flush()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["SPACY_FORCE_CPU"] = "true"


    global _worker_nlp, _worker_known_names
    # load spaCy and rulers *inside* the worker
    import spacy, pandas as pd
    _worker_nlp = spacy.load("sv_core_news_md", disable=["parser","lemmatizer"])

    df0 = pd.read_csv(known_names_csv)
    known_names = df0["name"].dropna().astype(str).unique().tolist()
    df1 = pd.read_csv(known_places_csv)
    known_places = df1["location"].dropna().astype(str).unique().tolist()

    add_known_names_ruler(_worker_nlp, known_names, known_places=known_places)
    _worker_known_names = known_names



    print(f"[PID {os.getpid()}] loading spacy...")
    _worker_nlp = spacy.load("sv_core_news_md", disable=["parser","lemmatizer"])
    print(f"[PID {os.getpid()}] reading CSVs...")
    df0 = pd.read_csv(known_names_csv)
    df1 = pd.read_csv(known_places_csv)
    print(f"[PID {os.getpid()}] init done")


def _get_worker_nlp():
    if _worker_nlp is None:
        raise RuntimeError("worker nlp not initialized")
    return _worker_nlp, _worker_known_names



def teardown_nlp():
    if hasattr(get_nlp, "_nlp"):
        del get_nlp._nlp
        del get_nlp._known_names
        gc.collect()
def process_one_star(arg):
    return process_one(arg)

def run_batch(paths, args, max_workers=10, task_timeout=60, batch_timeout=600):
    """
    Run one batch of XML files in parallel, returning (path, changed, err) for each.
    Never hangs: forcibly cleans up all workers on exit.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
    import psutil, traceback, gc, time, os, signal

    known_names_csv = "riksdagen-persons/data/name.csv"
    known_places_csv = "riksdagen-persons/data/location_specifier.csv"

    results = []
    stuck = []
    tasks = [(p, args) for p in paths]

    print(f"[PID {os.getpid()}] ⚙️ Starting run_batch for {len(tasks)} files")
    sys.stdout.flush()

    # -------- Global watchdog (batch-level) --------
    def timeout_handler(signum, frame):
        raise TimeoutError("⏰ batch timeout reached")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(batch_timeout)

    start_time = time.monotonic()

    # -------- Main execution --------
    ex = None
    try:
        ctx = mp.get_context("spawn")   # ✅ safer with spaCy
        ex = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=worker_init,
            initargs=(known_names_csv, known_places_csv),
            mp_context=ctx,
        )

        future_to_path = {ex.submit(process_one, t): t[0] for t in tasks}

        for idx, fut in enumerate(tqdm(as_completed(future_to_path, timeout=batch_timeout), total=len(future_to_path)), start=1):
            path = future_to_path[fut]
            print(path)
            try:

                res = fut.result(timeout=task_timeout)
                print("  --> OK")
            except TimeoutError:
                print(f"⚠ Timeout in worker: {path}")
                stuck.append(path)
                continue
            except Exception as e:
                tb = traceback.format_exc(limit=1)
                print(f"⚠ {path}: {e}")
                res = (path, False, f"processing failed: {e}\n{tb}")
            results.append(res)
            print(f"finished {idx} of {len(future_to_path)}")
            print(len(results))
    except TimeoutError as e:
        print(f"⏰ Batch timed out after {batch_timeout}s — aborting...")
        for path, _ in tasks:
            if path not in [r[0] for r in results]:
                results.append((path, False, "batch timeout"))
    finally:
        signal.alarm(0)
        print("🧹 Forcing executor shutdown...")
        if ex:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                print("shutdown failed")
        # -------- Kill any stray python worker processes --------
        time.sleep(0.5)
        for proc in psutil.process_iter():
            try:
                if proc.pid != os.getpid() and "python" in proc.name().lower():
                    if any(x in " ".join(proc.cmdline()) for x in ["refine-signature", "spacy", "ProcessPoolExecutor"]):
                        proc.terminate()
            except Exception:
                pass

        gc.collect()
        elapsed = time.monotonic() - start_time
        print(f"✅ Batch finished in {elapsed:.1f}s — {len(results)} results ({len(stuck)} stuck)\n")

    return results





def main(args):

    motions = list(args.motions)
    batch_size = 50
    batches = [motions[i:i+batch_size] for i in range(0, len(motions), batch_size)]

    for bi, chunk in enumerate(batches, 1):
        print(f"\n⚙️ Processing batch {bi}/{len(batches)} ({len(chunk)} files)")
        results = run_batch(chunk, args, max_workers=10, task_timeout=60)
        done, changed, failed = 0, 0, 0
        for path, ch, err in results:
            done += 1
            if err:
                failed += 1
            elif ch:
                changed += 1
        print(f"✅ Batch {bi} done: {done} files, {changed} changed, {failed} failed")


    """
    for bi, chunk in enumerate(batches, 1):
        print(f"⚙️ Batch {bi}/{len(batches)} ({len(chunk)} files)")
        with ctx.Pool(processes=procs, maxtasksperchild=tasks_per_child) as pool:
            for path, changed, err in pool.imap_unordered(process_one, [(m, args) for m in chunk], chunksize=1):
                if err:
                    print(f"⚠ {path}: {err}")
                elif changed:
                    print(f"✅ {path}")
                else:
                    print(f"⏩ {path}")
        print(f"✅ Batch {bi} finished")

    for bi, chunk in enumerate(batches, 1):
        print(f"⚙️  Processing batch {bi}/{len(batches)} — {len(chunk)} files")

        with closing(ctx.Pool(processes=procs, maxtasksperchild=tasks_per_child)) as pool:
            try:
                iterator = pool.imap_unordered(process_one, [(m, args) for m in chunk], chunksize=1)

                for path, changed, err in iterator:# tqdm(iterator, total=len(chunk), dynamic_ncols=True):
                    if err:
                        print(f"⚠ {path}: {err}")
                    elif changed:
                        print(f"✅ {path}")
                    else:
                        print(f"⏩ {path}")

                # Make sure all results are fully consumed *before* closing the pool
                pool.close()
                pool.join()

            except KeyboardInterrupt:
                print("⚠ Interrupted, terminating workers…")
                pool.terminate()
                pool.join()
            except Exception as e:
                print(f"⚠ Batch {bi} crashed: {e}")
                pool.terminate()
                pool.join()

        teardown_nlp()
        gc.collect()
    gc.collect()

    """


if __name__ == '__main__':
    parser = fetch_parser("motions", docstring=__doc__)
    parser.add_argument("--spacy", action='store_true', help="Run with spacy NER")
    main(impute_args(parser.parse_args()))
