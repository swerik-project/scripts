#!/usr/bin/env python3
import multiprocessing as mp

# Must come before any Pool, Executor, or spaCy import
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)
from multiprocessing import (
    get_context,
    Process,
)
from concurrent.futures import ProcessPoolExecutor
import gc
import lxml.etree as etree
import pandas as pd
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from pyriksdagen.io import (
    parse_tei,
    write_tei
)
import re
import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
import time
import tqdm as tqdm





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

def is_initial_token(t: str) -> bool:
    return bool(re.fullmatch(r"[A-ZÅÄÖ]\.?", t))

def is_name_word(t: str) -> bool:
    return bool(re.match(r"^[A-ZÅÄÖ][a-zåäö\-]+$", t))

NAME_WORD_RE = re.compile(rf"^[{UP}][{LO}\-]+$")

def is_word(tok: str) -> bool:
    return bool(NAME_WORD_RE.match(tok))

def looks_like_initials_plus_surname(toks, j) -> bool:
    """Return True if toks[j:] starts with 1+ initials followed by a name word."""
    i = j
    saw_initial = False
    while i < len(toks) and is_initial_token(toks[i]):
        saw_initial = True
        i += 1
    return saw_initial and i < len(toks) and is_name_word(toks[i])

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
    t = tok.strip().rstrip(":;,.!?")
    return bool(re.fullmatch(rf"[{UP}]\.?", t))


def is_initial_block(tok: str) -> bool:
    t = tok.strip().rstrip(":;,.!?")
    return bool(re.fullmatch(rf"(?:[{UP}]\.\s*){{1,3}}", t))


def is_hy_initial(tok: str) -> bool:
    t = tok.strip()
    return bool(re.fullmatch(rf"-[{UP}]\.?", t))


def is_name_token(tok: str) -> bool:
    return bool(NAME_TOKEN.match(tok))


def is_surname_like(tok: str) -> bool:
    t = tok.lower().rstrip(".")
    return any(t.endswith(s) for s in SURNAME_SUFFIXES)


if not Span.has_extension("is_known_name"):
    Span.set_extension("is_known_name", default=False)
if not Span.has_extension("is_known_place"):
    Span.set_extension("is_known_place", default=False)


class KnownNameRuler:
    def __init__(self, nlp, name, known_names, known_places):
        self.known_places = set(known_places)

        # Build a PhraseMatcher once (much faster than text.find in a loop)
        self.matcher = PhraseMatcher(nlp.vocab, attr="ORTH")
        # Only keep non-empty names
        patterns = [nlp.make_doc(n) for n in known_names if n and n.strip()]
        # You can shard patterns if extremely large; for now one label
        self.matcher.add("KNOWN_PERSON", patterns)

    def __call__(self, doc):
        new_ents = list(doc.ents)

        # Run the phrase matcher over the doc
        matches = self.matcher(doc)  # list of (match_id, start, end)
        for _, start, end in matches:
            span = doc[start:end]
            # Create a PER span (known person)
            span = doc.char_span(span.start_char, span.end_char, label="PER", alignment_mode="expand")
            if span is not None:
                span._.is_known_name = True
                new_ents.append(span)

        # Deduplicate & remove overlaps, preferring earlier & longer
        new_ents = sorted(new_ents, key=lambda s: (s.start_char, -s.end_char))
        filtered = []
        last_end = -1
        for ent in new_ents:
            if ent.start_char >= last_end:
                filtered.append(ent)
                last_end = ent.end_char
            # else skip overlaps

        doc.ents = tuple(filtered)
        return doc


@Language.factory("known_name_ruler")
def create_known_name_ruler(nlp, name, known_names, known_places):
    return KnownNameRuler(nlp, name, known_names, known_places)


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
        if "known_name_ruler" not in nlp.pipe_names:
            nlp.add_pipe(
                "known_name_ruler",
                before="ner",
                config={"known_names": known_names, "known_places": known_places},
            )
        return nlp, known_names, known_places


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
        if taken >= MAX_PLACE_TOKENS:
            break
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

def _clean_for_compare(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip(" \t\n.;:,!?")).strip()


def parse_signature_text(text, nlp, known_names, known_places):
    #print("parsing signature text")
    text = clean_text(text)
    #print(text)
    if not text:
        return []

    doc = nlp(text)
    toks, spans = _tokenize_with_spans(text)
    norm_toks = []
    for t in toks:
        core = re.sub(r"[:;,.!?]+$", "", t.strip())  # strip trailing punctuation for classification
        norm_toks.append(core if core else t)
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

    name_spans = [ent for ent in doc.ents if ent.label_ == "PER"]
    name_spans = sorted(name_spans, key=lambda e: e.start_char)
    loc_spans  = [ent for ent in doc.ents if ent.label_ in ("LOC", "GPE")]
    # index entities by start offset for quick lookup
    ent_starts = {e.start_char: e for e in name_spans}
    loc_starts = {e.start_char: e for e in loc_spans}

    print(f"TOKENS ({len(toks)}): {toks}")

    MAX_GLOBAL_STEPS = 5000
    global_steps = 0
    while i < len(toks):
        global_steps += 1
        if global_steps > MAX_GLOBAL_STEPS:
            print(f"🚨 Safety break: exceeded {MAX_GLOBAL_STEPS} steps")
            break
        if global_steps % 1000 == 0:
            print(f"[{global_steps}] i={i}/{len(toks)} tok={toks[i]!r}")
        #print("--", i, len(toks))
        i_prev=i
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
        norm_tok = norm_toks[i]

        # if re.match(r".*[:;,.!?]$", tok):
        #     core = tok.rstrip(":;,.!?")
        #     toks[i] = core
        #     if core and re.match(r"^[A-ZÅÄÖ]", core):
        #         tok = core


        #print("t", toks)
        #print("o", out)
        # ---- record id -------------------------------------------------------
        two = " ".join(norm_toks[i:i+2])
        if re.match(r"(?i)\bN[:r]\s*\d+\b", two):
            #print("  -> D-bug: match record ID")
            out.append({"type": "record-id", "text": two})
            i += 2
            continue

        # ---- location specifier ---------------------------------------------
        elif norm_tok.lower() in {"i", "från"}:
            #print("  -> D-bug: match i-ort")
            j = i + 1

            # Try to consume a following proper noun or entity
            if j < len(toks):
                nxt = norm_toks[j]
                if re.match(r"^[A-ZÅÄÖ][a-zåäö\-]+$", nxt):
                    # e.g. "i Dahl", "från Stockholm"
                    j += 1
                elif j + 1 < len(toks):
                    # handle two-part place names like "i Nya Kopparberg"
                    nxt2 = norm_toks[j + 1]
                    if re.match(r"^[A-ZÅÄÖ][a-zåäö\-]+$", nxt) and re.match(r"^[A-ZÅÄÖ][a-zåäö\-]+$", nxt2):
                        j += 2

            frag = " ".join(toks[i:j]).strip()
            out.append({"type": "location-specifier", "text": frag})
            i = j
            continue

        # ---- matched entity -------------------------------------------------
        elif s in ent_starts:
            #print("  -> D-bug: match ent")
            ent = ent_starts[s]
            out.append({"type": "person", "text": ent.text})
            # advance i safely past the entity span
            j = i
            while j < len(spans) and spans[j][2] <= ent.end_char:
                j += 1
            i = j
            continue

        # Skip tokens that end with punctuation unless clearly part of a name
        elif re.match(r".*[.:;!?]$", tok) and not is_initial(norm_tok) and not is_name_word(norm_tok.rstrip(":;.!?")):
            out.append({"type": "other", "text": tok})
            i += 1
            continue

        elif re.fullmatch(r"[,;]", norm_tok):
            i += 1
            continue

        elif re.match(r"^[A-ZÅÄÖ]", norm_tok) and norm_tok.lower() not in {"på", "av", "och", "från"}:
            #print("  -> D-bug: match initials/name group")
            j = i
            name_tokens = []
            inner_guard = 0
            MAX_INNER_GUARD = 100
            while j < len(toks):
                inner_guard += 1
                j_prev = j
                nxt = norm_toks[j]

                # Hard stop markers
                if nxt.lower() in {"i", "från", "på", "av"} or nxt in {"Nr", "nr"}:
                    break

                # --- Rule 0e: standalone colon or dash → consume and end current name
                if nxt in {":", "-", "–", "—"}:
                    j += 1
                    continue

                # --- Rule 0a: break on obviously lowercase or non-name words
                if re.fullmatch(r"[a-zåäö\-]+", nxt):
                    break

                # --- Rule 0b: nobiliary particles continue the current name
                if nxt.lower() in NOBILIARY_PARTICLES:
                    name_tokens.append(toks[j])
                    j += 1
                    continue

                # --- Rule 0c: handle multi-letter pseudo-initials like 'AA.' or 'ÅA.'
                if re.fullmatch(r"([A-ZÅÄÖ]{2,3}\.?)", nxt):
                    name_tokens.append(toks[j])
                    j += 1
                    continue

                # --- Rule 0d: skip trivial punctuation
                if re.fullmatch(r"[,;]", nxt):
                    j += 1
                    while j < len(toks) and toks[j].islower():
                        j += 1
                    break

                # Accept initials or capitalized words
                if is_initial_block(nxt) or is_initial(nxt) or is_word(nxt) or re.fullmatch(r"[A-ZÅÄÖ][a-zåäö]{1,3}\.", nxt):
                    name_tokens.append(toks[j])
                    j += 1

                    # --- Rule 1: initials + word followed by another initial → new name
                    if (
                        is_initial(name_tokens[-1])
                        and j < len(toks)
                        and (is_initial(norm_toks[j]) or is_word(norm_toks[j]))
                    ):
                        continue

                    # --- Rule 2: don’t end on an initial; continue until surname-like word
                    if j < len(toks) and is_initial(name_tokens[-1]) and not is_word(norm_toks[j]):
                        continue

                    # --- Rule 2b: no discontiguous initials
                    if (
                        len(name_tokens) >= 2
                        and is_initial(name_tokens[-1])
                        and not is_initial(name_tokens[-2])
                        and j < len(toks)
                        and is_initial(norm_toks[j])
                    ):
                        break

                    # --- Rule 2c: don't end on bare initials unless followed by surname-like word or nobiliary particle
                    if (
                        all(is_initial(t) for t in name_tokens)
                        and j < len(toks)
                        and norm_toks[j].lower() not in NOBILIARY_PARTICLES
                        and not is_word(norm_toks[j])
                    ):
                        break

                    # --- Rule 3: detect next full-name start (Firstname + NextWord)
                    if (
                        len(name_tokens) >= 2
                        and j < len(toks)
                        and is_word(norm_toks[j])
                        and (j + 1 < len(toks) and (is_word(norm_toks[j + 1]) or is_initial(norm_toks[j + 1])))
                    ):
                        break

                    # --- Rule 4b: stop if we've already got a surname-like word
                    # and the next token is just an initial (to prevent 'Ahlberga P.')
                    if (
                        len(name_tokens) >= 1
                        and is_word(name_tokens[-1])
                        and j < len(toks)
                        and is_initial(norm_toks[j])
                        and norm_toks[j].lower() not in NOBILIARY_PARTICLES
                    ):
                        break

                    # --- Rule 5: if next tokens look like a *new* initials+surname combo, split here
                    if (
                        len(name_tokens) >= 2
                        and looks_like_initials_plus_surname(norm_toks, j)
                    ):
                        break

                    continue

                # --- Rule 6: if next token(s) form a known name, break here
                lookahead = " ".join(norm_toks[j:j+3])
                found_known = False
                for kname in known_names:
                    if lookahead.startswith(kname) or norm_toks[j] in kname.split():
                        found_known = True
                        break
                if found_known:
                    break  # break the inner name loop so outer loop can start next name at j

                # --- safety: ensure progress even on punctuation or noise
                # --- safety: ensure progress even on punctuation or noise
                if j == j_prev:
                    print(f"⚠️ Inner name-loop stuck at {j}, token={toks[j]!r}")
                    i = j + 1
                    break

                if inner_guard > MAX_INNER_GUARD:
                    print(f"🚨 Inner loop emergency break at token {j}, {toks[j]!r}")
                    i = j + 1   # advance outer index so we don’t re-enter
                    break

                   # --- finalize name tokens -----------------------------------------
                    frag = " ".join(name_tokens).strip()
                    if frag:
                        cur  = _clean_for_compare(frag)
                        last = _clean_for_compare(out[-1]["text"]) if out else None
                        if not out or cur != last:
                            out.append({"type": "person", "text": frag})

                    # --- always advance safely ----------------------------------------
                    # 1.  normally, go to j
                    # 2.  if punctuation-ended, also skip that token
                    if j >= len(toks):
                        break
                    if toks[j - 1].endswith((':', ';', '.')):
                        i = j        # skip the punctuation token
                    else:
                        i = max(i + 1, j)

                    continue

                # ----------------------------------------------------------------------
                # catch-all for tokens that weren't matched by any rule
                out.append({"type": "other", "text": tok})
                i += 1
                continue


            # --- Strong guard to prevent Wallin-type repetition ---
            # If we hit the end of text or the token was punctuation-terminated, stop processing this item
            if j >= len(toks):
                break  # end of tokens reached safely

            # Defensive: check the current token only if still within range
            if i < len(toks) and toks[i].endswith((':', ';', '.')):
                i = j + 1
                continue

            # If nothing consumed, force skip
            if j == i:
                if i < len(toks):
                    print(f"⚠️ No progress / stuck on token {i}, tok={toks[i]!r} — skipping")
                i += 1
                continue

            # Normal advance
            i = max(i, j)
            continue

        else:
            # Catch-all: preserve unclassified token(s), avoid duplicates (ignore punctuation)
            cur = _clean_for_compare(tok)
            last = _clean_for_compare(out[-1]["text"]) if out else None
            if not out or cur != last:
                out.append({"type": "other", "text": tok})
            i += 1
            continue
        # ---- end of main while i < len(toks) iteration ----
        if i_prev == i:
            print(f"⚠️ No progress at token index {i}, tok={toks[i]!r} — forcing advance")
            cur = _clean_for_compare(toks[i])
            last = _clean_for_compare(out[-1]["text"]) if out else None
            if not out or cur != last:
                out.append({"type": "other", "text": toks[i]})
            i += 1
            continue

    del doc
    #print("out --", out)
    return out


def looks_like_signature_block(sig_block, nlp):
    #print("is it a signature block?")
    texts = " ".join([l.strip() for itm in sig_block.itertext() for l in itm.splitlines() if l is not None and l.strip()!=''])
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

    if len(words) > 100:
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


def expand_signatures(root, ns, nlp, known_names, known_places):
    #print("expanding signatures")
    changed = False
    signature_blocks = root.findall(f".//{ns['tei_ns']}div[@type=\"signatureBlock\"]")
    for sb in signature_blocks:
        if not looks_like_signature_block(sb, nlp):
            continue
        lists = sb.findall(f".//{ns['tei_ns']}list")
        for list_ in lists:
            old_items = list(list_.findall(f"{ns['tei_ns']}item"))
            if not old_items:
                continue

            full_text = ' '.join([l.strip() for t in list_.itertext() for l in t.splitlines() if l.strip() != ''])

            if not full_text or full_text=='':
                continue

            parsed = parse_signature_text(full_text, nlp, known_names, known_places)

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

            list_[:] = new_items
            changed = True
    return changed


def parse_signature_block(mots, start=1):
    #print(f"    parsing signature block")
    nlp, known_names, known_places = _load_spacy()
    for i, mot in enumerate(mots, start):
        print("  ", i, mot)
        root,ns = parse_tei(mot)
        changed = expand_signatures(root, ns, nlp, known_names, known_places)
        if changed:
            write_tei(root, mot)
        del root
        del ns


def run_batch(batch, start_i, max_workers=10):
    print(f"... run batch with {max_workers} processes")
    ctx = mp.get_context("spawn")   # ✅ safer with spaCy
    ex = ProcessPoolExecutor(
        max_workers=max_workers,
        #initializer=worker_init,
        #initargs=(known_names_csv, known_places_csv),
        mp_context=ctx,
    )
    ex.submit(parse_signature_block, batch, start_i)




def main(args):
    args.motions = [m for m in args.motions if not m.endswith("-fört.xml") and not m.endswith("-reg.xml")]
    if args.use_multithreading:
        print("Using multithread")
        batches = [args.motions[i:i+args.batch_size] for i in range(0, len(args.motions), args.batch_size)]

        for bi, batch in enumerate(batches, 1):
            print(f"\n\n starting batch {bi} of {len(batches)}\n\n")
            result = run_batch(batch, bi*args.batch_size, max_workers=args.n_workers)
            gc.collect()
    else:
        parse_signature_block(args.motions)




if __name__ == '__main__':
    parser = fetch_parser("motions", docstring=__doc__)
    parser.add_argument("--use-multithreading", action='store_true')
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--n-workers", type=int, default=10)
    main(impute_args(parser.parse_args()))

