#!/usr/bin/env python3
"""
Propose high-confidence fixes for truncated motion signature blocks.

The script looks for immediate paragraphs after an existing <signatureBlock>
that are made up of active MP names, party labels, and location specifiers.
Dry-run is the default. Use --apply to append new signature items to the
signature block and remove the consumed paragraphs.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from lxml import etree
from pyriksdagen.args import fetch_parser, impute_args
from pyriksdagen.io import TEI_NS, parse_tei, write_tei
from pyriksdagen.utils import get_formatted_uuid, infer_metadata
from trainerlog import get_logger


LOGGER = get_logger("signature-block-extension-proposals")
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
STOP_WORDS = {
    "av",
    "bilaga",
    "gotab",
    "hemstall",
    "hemstalles",
    "kungl",
    "motion",
    "motionen",
    "motioner",
    "proposition",
    "protokoll",
    "riksdag",
    "riksdagen",
    "yrkar",
}
PARTY_LABELS = {
    "c",
    "cp",
    "fp",
    "h",
    "kd",
    "kds",
    "l",
    "m",
    "mp",
    "nyd",
    "s",
    "sd",
    "v",
    "vpk",
}
PARTY_LABEL_RE = re.compile(
    r"\(\s*(" + "|".join(sorted(PARTY_LABELS, key=len, reverse=True)) + r")\s*\)",
    flags=re.IGNORECASE,
)
MOTION_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:Mot\.?\s*)?(?:\d{4}/\d{2}\s*)?(?:[A-Za-zÅÄÖåäö]{1,5}\s*)?0?\d{1,4}\s+",
    flags=re.IGNORECASE,
)
LABEL_ONLY_SIGNATURE_RE = re.compile(
    r"^\s*(?:Mot\.?\s*)?(?:\d{4}/\d{2})?(?:\s*[A-Za-zÅÄÖåäö]{1,5})?\s*\d*(?:[!:]\d+)?\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Token:
    norm: str
    raw: str
    start: int
    end: int


@dataclass(frozen=True)
class NameCandidate:
    person_id: str
    display_name: str
    tokens: tuple[str, ...]
    parties: frozenset[str]
    locations: frozenset[str]


@dataclass(frozen=True)
class NameMatch:
    person_id: str
    display_name: str
    token_start: int
    token_end: int
    start: int
    end: int
    match_kind: str = "exact"
    repair_note: str = ""


@dataclass(frozen=True)
class SignatureItem:
    person_id: str
    text: str
    match_kind: str
    repair_note: str
    item_id: str = ""


@dataclass(frozen=True)
class Proposal:
    motion: str
    parliament_year: str
    calendar_year: int
    chamber: str
    signature_block_id: str
    list_id: str
    paragraph_id: str
    paragraph_text: str
    add_items: tuple[SignatureItem, ...]
    update_items: tuple[SignatureItem, ...] = tuple()


class PersonIndex:
    def __init__(self, persons_root: Path):
        self.names_by_person: dict[str, set[str]] = defaultdict(set)
        self.locations_by_person: dict[str, set[str]] = defaultdict(set)
        self.mp_rows: list[dict[str, str]] = []
        self.party_rows: list[dict[str, str]] = []
        self.party_to_abbrevs: dict[str, set[str]] = defaultdict(set)
        self._active_cache: dict[tuple[int, str], list[NameCandidate]] = {}
        self._load(persons_root)

    def _load(self, persons_root: Path) -> None:
        for row in read_csv(persons_root / "data" / "name.csv"):
            name = row.get("name", "").strip()
            if name:
                self.names_by_person[row["person_id"]].add(name)

        for row in read_csv(persons_root / "data" / "location_specifier.csv"):
            location = row.get("location", "").strip()
            if location:
                self.locations_by_person[row["person_id"]].add(location)

        self.mp_rows = read_csv(persons_root / "data" / "member_of_parliament.csv")
        self.party_rows = read_csv(persons_root / "data" / "party_affiliation.csv")
        for row in read_csv(persons_root / "data" / "party_abbreviation.csv"):
            party = normalize(row.get("party", ""))
            abbreviation = normalize(row.get("abbreviation", ""))
            if party and abbreviation:
                self.party_to_abbrevs[party].add(abbreviation)

    def active_candidates(self, year: int, chamber: str) -> list[NameCandidate]:
        key = (year, chamber)
        if key in self._active_cache:
            return self._active_cache[key]

        active_people = {
            row["person_id"]
            for row in self.mp_rows
            if overlaps_year(row.get("start"), row.get("end"), year)
            and chamber_matches(row.get("role", ""), chamber)
        }
        parties_by_person: dict[str, set[str]] = defaultdict(set)
        for row in self.party_rows:
            person_id = row["person_id"]
            if person_id not in active_people or not overlaps_year(row.get("start"), row.get("end"), year):
                continue
            party_norm = normalize(row.get("party", ""))
            if party_norm:
                parties_by_person[person_id].add(party_norm)
                parties_by_person[person_id].update(self.party_to_abbrevs.get(party_norm, set()))

        candidates = []
        seen = set()
        for person_id in sorted(active_people):
            for name in sorted(self.names_by_person.get(person_id, set())):
                tokens = tuple(tokenize(name))
                if len(tokens) < 2:
                    continue
                dedupe_key = (person_id, tokens)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                candidates.append(NameCandidate(
                    person_id=person_id,
                    display_name=name,
                    tokens=tokens,
                    parties=frozenset(parties_by_person.get(person_id, set())),
                    locations=frozenset(normalize(loc) for loc in self.locations_by_person.get(person_id, set())),
                ))
        self._active_cache[key] = candidates
        return candidates


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as inf:
        return list(csv.DictReader(inf))


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [token for token in normalize(text).split() if token]


def token_spans(text: str) -> list[Token]:
    tokens = []
    for match in re.finditer(r"[^\W_]+", text, flags=re.UNICODE):
        raw = match.group(0)
        norm = normalize(raw)
        if norm:
            tokens.append(Token(norm=norm, raw=raw, start=match.start(), end=match.end()))
    return tokens


def text_content(elem) -> str:
    return " ".join(t.strip() for t in elem.itertext() if t and t.strip())


def year_from_date(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"(18|19|20)\d{2}", str(value))
    return int(match.group(0)) if match else None


def overlaps_year(start: str | None, end: str | None, year: int) -> bool:
    start_year = year_from_date(start)
    end_year = year_from_date(end)
    if start_year is not None and start_year > year:
        return False
    if end_year is not None and end_year < year:
        return False
    return True


def chamber_matches(role: str, chamber: str) -> bool:
    role = (role or "").lower()
    if chamber == "Första kammaren":
        return "förstakammar" in role or role == "ledamot"
    if chamber == "Andra kammaren":
        return "andrakammar" in role or role == "ledamot"
    return True


def is_boundary_or_body_text(elem, motion_number: int | None) -> bool:
    if elem.tag.rsplit("}", 1)[-1] != "p":
        return True
    if elem.get("type"):
        return True
    text = text_content(elem)
    norm = normalize(text)
    if not norm:
        return True
    if re.match(r"^n\s*o\s+\d+", norm) or re.match(r"^nr\s+\d+", norm):
        return True
    if motion_number is not None and re.match(rf"^(n\s*o|nr)\s+0*{motion_number}\b", norm):
        return True
    if norm.startswith("motioner i ") or norm.startswith("motion till riksdagen"):
        return True
    return False


def party_tokens_near(text: str) -> set[str]:
    return {normalize(match.group(1)) for match in re.finditer(r"\(([^)]+)\)", text)}


def is_location_marker_token(token: str) -> bool:
    return token in {"i", "fran"}


def edit_distance_at_most_one(left: str, right: str) -> bool:
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) == 1
    if len(left) > len(right):
        left, right = right, left
    i = j = edits = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        j += 1
    return True


def tiny_ocr_repair_note(expected: str, observed: str) -> str | None:
    if expected == observed:
        return ""
    if len(expected) < 4 or len(observed) < 3:
        return None
    if observed == expected[1:]:
        return f"missing-leading-char:{observed}->{expected}"
    if expected == observed[1:]:
        return f"extra-leading-char:{observed}->{expected}"
    if edit_distance_at_most_one(expected, observed):
        return f"edit-distance-1:{observed}->{expected}"
    return None


def candidate_ocr_repair_note(tokens: list[Token], start: int, candidate: NameCandidate) -> str | None:
    n = len(candidate.tokens)
    observed = [token.norm for token in tokens[start:start + n]]
    expected = list(candidate.tokens)
    if len(observed) != n or n < 2:
        return None
    if observed[-1] != expected[-1]:
        return None

    notes = []
    for observed_token, expected_token in zip(observed[:-1], expected[:-1]):
        note = tiny_ocr_repair_note(expected_token, observed_token)
        if note is None:
            return None
        if note:
            notes.append(note)
    if len(notes) != 1:
        return None
    return notes[0]


def find_name_matches(text: str, candidates: list[NameCandidate]) -> list[NameMatch]:
    tokens = token_spans(text)
    by_first: dict[str, list[NameCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_first[candidate.tokens[0]].append(candidate)

    matches = []
    for i, token in enumerate(tokens):
        possible = []
        for candidate in by_first.get(token.norm, []):
            n = len(candidate.tokens)
            if tuple(t.norm for t in tokens[i:i + n]) == candidate.tokens:
                possible.append(candidate)
        repair_notes: dict[tuple[str, tuple[str, ...]], str] = {}
        match_kind = "exact"
        if not possible:
            for candidate in candidates:
                n = len(candidate.tokens)
                if i + n > len(tokens):
                    continue
                repair_note = candidate_ocr_repair_note(tokens, i, candidate)
                if repair_note is None:
                    continue
                possible.append(candidate)
                repair_notes[(candidate.person_id, candidate.tokens)] = repair_note
            match_kind = "ocr-repair"
        if not possible:
            continue

        longest = max(len(candidate.tokens) for candidate in possible)
        possible = [candidate for candidate in possible if len(candidate.tokens) == longest]
        if len({candidate.person_id for candidate in possible}) != 1:
            continue
        candidate = possible[0]
        repair_note = repair_notes.get((candidate.person_id, candidate.tokens), "")
        matches.append(NameMatch(
            person_id=candidate.person_id,
            display_name=candidate.display_name,
            token_start=i,
            token_end=i + len(candidate.tokens),
            start=tokens[i].start,
            end=tokens[i + len(candidate.tokens) - 1].end,
            match_kind=match_kind,
            repair_note=repair_note,
        ))

    selected = []
    last_end = -1
    for match in sorted(matches, key=lambda m: (m.start, -(m.end - m.start))):
        if match.start < last_end:
            continue
        selected.append(match)
        last_end = match.end
    return selected


def existing_signature_people(block, ns: dict[str, str]) -> set[str]:
    people = set()
    for item in block.findall(f".//{ns['tei_ns']}item[@type='signature']"):
        who = item.get("who")
        if who and who != "unknown":
            people.add(who)
    return people


def segment_is_safe(segment: str, match: NameMatch, candidate_by_person: dict[str, NameCandidate]) -> bool:
    if len(segment) > 140:
        return False
    norm = normalize(segment)
    if STOP_WORDS & set(norm.split()):
        return False
    candidate = candidate_by_person[match.person_id]
    # Party labels in the person database are not complete enough to make
    # printed abbreviations such as (m) and (fp) a hard validation rule here.
    # The high-confidence gate is the unique active MP-name match.
    if match.match_kind == "ocr-repair":
        parties = party_tokens_near(segment)
        if not parties:
            return False
        if candidate.parties and not parties & candidate.parties:
            return False
    location_markers = re.findall(r"\b(?:i|från|fran)\s+[A-ZÅÄÖ]", segment)
    if len(location_markers) > 1:
        return False
    if not segment[: match.end - match.start + 12].strip():
        return False
    return True


def token_indices_covered(matches: list[NameMatch]) -> set[int]:
    covered = set()
    for match in matches:
        covered.update(range(match.token_start, match.token_end))
    return covered


def token_is_parenthesized_party_label(text: str, token: Token) -> bool:
    before = text[:token.start].rstrip()
    after = text[token.end:].lstrip()
    return token.norm in PARTY_LABELS and before.endswith("(") and after.startswith(")")


def safe_between_tokens(tokens: list[Token], start: int, end: int, covered: set[int], text: str) -> bool:
    for index, token in enumerate(tokens[start:end], start=start):
        if index in covered:
            continue
        if token_is_parenthesized_party_label(text, token):
            continue
        if token.norm in {"i", "fran"}:
            continue
        return False
    return True


def find_location_spans(tokens: list[Token], matches: list[NameMatch], text: str) -> list[tuple[int, int, str]]:
    match_starts = {match.token_start for match in matches}
    spans = []
    i = 0
    while i < len(tokens):
        if not is_location_marker_token(tokens[i].norm):
            i += 1
            continue
        j = i + 1
        while j < len(tokens) and j not in match_starts and tokens[j].norm not in PARTY_LABELS and not is_location_marker_token(tokens[j].norm):
            j += 1
        if j > i + 1:
            spans.append((i, j, text[tokens[i].start:tokens[j - 1].end]))
        i = max(j, i + 1)
    return spans


def fallback_single_match_location_is_safe(raw_location: str) -> bool:
    if PARTY_LABEL_RE.search(raw_location):
        return False
    location_text = re.sub(r"^(i|från|fran)\s+", "", raw_location, flags=re.IGNORECASE).strip()
    if re.search(r"\.\s+\S", location_text):
        return False
    tokens = token_spans(location_text)
    if not 1 <= len(tokens) <= 2:
        return False
    for token in tokens:
        if token.norm in PARTY_LABELS or token.raw.isdigit():
            return False
        if len(token.norm) == 1:
            return False
        if not token.raw[:1].isupper():
            return False
    return True


def assign_location_spans(
    spans: list[tuple[int, int, str]],
    matches: list[NameMatch],
    candidate_by_person: dict[str, NameCandidate],
) -> tuple[dict[int, str], set[int]] | None:
    assignments: dict[int, list[str]] = defaultdict(list)
    covered_tokens: set[int] = set()
    for span_start, span_end, raw_location in spans:
        loc_value = normalize(re.sub(r"^(i|från|fran)\s+", "", raw_location, flags=re.IGNORECASE))
        possible = []
        for index, match in enumerate(matches):
            candidate = candidate_by_person[match.person_id]
            if loc_value in candidate.locations:
                distance = min(abs(span_start - match.token_end), abs(match.token_start - span_end))
                possible.append((distance, index))
        if not possible:
            if len(matches) == 1 and fallback_single_match_location_is_safe(raw_location):
                assignments[0].append(raw_location)
                covered_tokens.update(range(span_start, span_end))
                continue
            return None
        possible.sort()
        if len(possible) > 1 and possible[0][0] == possible[1][0]:
            return None
        assignments[possible[0][1]].append(raw_location)
        covered_tokens.update(range(span_start, span_end))
    return {index: " ".join(values) for index, values in assignments.items()}, covered_tokens


def paragraph_has_unconsumed_content(text: str, matches: list[NameMatch], covered_locations: set[int]) -> bool:
    tokens = token_spans(text)
    covered = token_indices_covered(matches)
    covered.update(covered_locations)
    for index, token in enumerate(tokens):
        if index in covered:
            continue
        if token.norm in PARTY_LABELS or token.norm in {"i", "fran"}:
            continue
        if token.raw.isdigit():
            continue
        return True
    return False


def paragraph_has_unconsumed_strict_content(text: str, matches: list[NameMatch], covered_locations: set[int]) -> bool:
    tokens = token_spans(text)
    covered = token_indices_covered(matches)
    covered.update(covered_locations)
    return any(index not in covered for index, _ in enumerate(tokens))


def safe_between_tokens_strict(tokens: list[Token], start: int, end: int, covered: set[int]) -> bool:
    return all(index in covered for index in range(start, end))


def abbreviated_token_matches(token: Token, candidate_token: str) -> bool:
    if token.norm == candidate_token:
        return True
    if len(token.norm) >= 2 and candidate_token.startswith(token.norm):
        return True
    if len(token.norm) <= 2 and token.raw[:1].isupper() and candidate_token[:1] in token.norm:
        return True
    return False


def candidate_matches_observed_name_tokens(tokens: list[Token], candidate: NameCandidate) -> bool:
    if len(tokens) < 2 or len(tokens) > len(candidate.tokens):
        return False
    if tokens[-1].norm != candidate.tokens[-1]:
        return False
    observed_given = tokens[:-1]
    candidate_given = candidate.tokens[:-1]
    if len(observed_given) > len(candidate_given):
        return False
    return all(
        abbreviated_token_matches(observed, expected)
        for observed, expected in zip(observed_given, candidate_given)
    )


def find_name_matches_with_abbreviations(text: str, candidates: list[NameCandidate]) -> list[NameMatch]:
    tokens = token_spans(text)
    matches = list(find_name_matches(text, candidates))
    for start in range(len(tokens)):
        for end in range(start + 2, min(len(tokens), start + 6) + 1):
            observed = tokens[start:end]
            possible = [
                candidate
                for candidate in candidates
                if candidate_matches_observed_name_tokens(observed, candidate)
            ]
            if not possible:
                continue
            if tuple(token.norm for token in observed) in {candidate.tokens for candidate in possible}:
                continue
            if len({candidate.person_id for candidate in possible}) != 1:
                continue
            candidate = possible[0]
            matches.append(NameMatch(
                person_id=candidate.person_id,
                display_name=candidate.display_name,
                token_start=start,
                token_end=end,
                start=observed[0].start,
                end=observed[-1].end,
                match_kind="abbrev-prefix",
                repair_note="abbreviated-name-continuation",
            ))

    selected = []
    last_end = -1
    for match in sorted(matches, key=lambda m: (m.start, -(m.end - m.start), m.match_kind != "exact")):
        if match.start < last_end:
            continue
        selected.append(match)
        last_end = match.end
    return selected


def paragraph_to_items(text: str, candidates: list[NameCandidate], existing_people: set[str]) -> tuple[SignatureItem, ...]:
    if len(text) > 700:
        return tuple()
    if STOP_WORDS & set(tokenize(text)):
        return tuple()

    matches = find_name_matches(text, candidates)
    if not matches:
        return tuple()

    prefix = text[:matches[0].start]
    if tokenize(prefix):
        return tuple()

    candidate_by_person = {}
    for candidate in candidates:
        if candidate.person_id not in candidate_by_person:
            candidate_by_person[candidate.person_id] = candidate

    tokens = token_spans(text)
    location_spans = find_location_spans(tokens, matches, text)
    location_assignment = assign_location_spans(location_spans, matches, candidate_by_person)
    if location_assignment is None:
        return tuple()
    assigned_locations, covered_location_tokens = location_assignment
    for index, match in enumerate(matches[:-1]):
        if not safe_between_tokens_strict(tokens, match.token_end, matches[index + 1].token_start, covered_location_tokens):
            return tuple()
    if paragraph_has_unconsumed_strict_content(text, matches, covered_location_tokens):
        return tuple()

    items = []
    seen_people = set(existing_people)
    for index, match in enumerate(matches):
        if match.person_id in seen_people:
            continue
        next_start = matches[index + 1].start if index + 1 < len(matches) else len(text)
        segment_end = next_start
        for span_start, _, _ in location_spans:
            if match.token_end <= span_start and (index + 1 == len(matches) or span_start < matches[index + 1].token_start):
                segment_end = tokens[span_start].start
                break
        segment = re.sub(r"\s+", " ", text[match.start:segment_end]).strip(" ,;[]-=—–")
        if index in assigned_locations:
            segment = f"{segment} {assigned_locations[index]}"
        if not segment_is_safe(segment, match, candidate_by_person):
            return tuple()
        items.append(SignatureItem(
            person_id=match.person_id,
            text=segment,
            match_kind=match.match_kind,
            repair_note=match.repair_note,
        ))
        seen_people.add(match.person_id)

    if not items:
        return tuple()
    return tuple(items)


def paragraph_to_abbreviated_continuation_items(
    text: str,
    candidates: list[NameCandidate],
    existing_people: set[str],
    proposals: list[Proposal],
    block_id: str,
    candidate_by_person: dict[str, NameCandidate],
) -> tuple[SignatureItem, ...]:
    if len(text) > 700:
        return tuple()
    if STOP_WORDS & set(tokenize(text)):
        return tuple()

    matches = find_name_matches_with_abbreviations(text, candidates)
    if not matches:
        return tuple()

    tokens = token_spans(text)
    leading_covered_tokens: set[int] = set()
    scratch_proposals = list(proposals)
    prefix = text[:matches[0].start]
    if tokenize(prefix):
        bare_location = bare_location_only_text(prefix, candidates)
        if bare_location is None:
            return tuple()
        if not attach_bare_location_to_pending_item(
            scratch_proposals,
            block_id,
            bare_location,
            candidate_by_person,
        ):
            return tuple()
        leading_covered_tokens.update(
            index for index, token in enumerate(tokens) if token.end <= matches[0].start
        )

    for candidate in candidates:
        if candidate.person_id not in candidate_by_person:
            candidate_by_person[candidate.person_id] = candidate

    location_spans = find_location_spans(tokens, matches, text)
    location_assignment = assign_location_spans(location_spans, matches, candidate_by_person)
    if location_assignment is None:
        return tuple()
    assigned_locations, covered_location_tokens = location_assignment
    covered_location_tokens.update(leading_covered_tokens)

    for index, match in enumerate(matches[:-1]):
        if not safe_between_tokens(tokens, match.token_end, matches[index + 1].token_start, covered_location_tokens, text):
            return tuple()
    if paragraph_has_unconsumed_content(text, matches, covered_location_tokens):
        return tuple()

    items = []
    seen_people = set(existing_people)
    for index, match in enumerate(matches):
        if match.person_id in seen_people:
            continue
        next_start = matches[index + 1].start if index + 1 < len(matches) else len(text)
        segment_end = next_start
        for span_start, _, _ in location_spans:
            if match.token_end <= span_start and (index + 1 == len(matches) or span_start < matches[index + 1].token_start):
                segment_end = tokens[span_start].start
                break
        segment = re.sub(r"\s+", " ", text[match.start:segment_end]).strip(" ,;[]-=—–")
        if index in assigned_locations:
            segment = f"{segment} {assigned_locations[index]}"
        if not segment_is_safe(segment, match, candidate_by_person):
            return tuple()
        items.append(SignatureItem(
            person_id=match.person_id,
            text=segment,
            match_kind=match.match_kind,
            repair_note=match.repair_note,
        ))
        seen_people.add(match.person_id)

    if not items:
        return tuple()
    proposals[:] = scratch_proposals
    return tuple(items)


def has_pending_signature_items(proposals: list[Proposal], block_id: str) -> bool:
    return any(proposal.signature_block_id == block_id and proposal.add_items for proposal in proposals)


def paragraph_to_single_name_continuation_item(
    text: str,
    candidates: list[NameCandidate],
    existing_people: set[str],
) -> tuple[SignatureItem, ...]:
    if len(text) > 120:
        return tuple()
    if STOP_WORDS & set(tokenize(text)):
        return tuple()
    if re.search(r"\([^)]*\)", text):
        return tuple()

    cleaned = re.sub(r"\s+", " ", text).strip(" ,;[]-=—–")
    if not single_name_punctuation_is_safe(cleaned):
        return tuple()
    if not signature_name_prefix_is_safe(cleaned):
        return tuple()

    tokens = token_spans(cleaned)
    if any(is_location_marker_token(token.norm) or token.norm in PARTY_LABELS for token in tokens):
        return tuple()
    observed = tuple(token.norm for token in tokens)

    possible = [
        candidate
        for candidate in candidates
        if len(candidate.tokens) > len(observed)
        and candidate.tokens[:len(observed)] == observed
    ]
    possible_people = {candidate.person_id for candidate in possible}
    if possible_people & existing_people:
        return tuple()
    if len(possible_people) == 1:
        return (SignatureItem(
            person_id=next(iter(possible_people)),
            text=cleaned,
            match_kind="unique-prefix",
            repair_note="single-name-continuation-prefix",
        ),)

    return tuple()


def single_name_punctuation_is_safe(text: str) -> bool:
    if re.search(r"[,;:]", text):
        return False
    for match in re.finditer(r"\.", text):
        if match.end() == len(text):
            continue
        prefix = text[:match.start()].rstrip()
        previous = re.search(r"([^\s.]+)$", prefix)
        if previous is None:
            return False
        token = previous.group(1)
        if len(token) != 1 or not token[:1].isupper():
            return False
    return True


def paragraph_to_items_after_leading_location(
    text: str,
    candidates: list[NameCandidate],
    existing_people: set[str],
    proposals: list[Proposal],
    block,
    ns: dict[str, str],
    block_id: str,
    candidate_by_person: dict[str, NameCandidate],
) -> tuple[tuple[SignatureItem, ...], tuple[SignatureItem, ...]]:
    matches = find_name_matches(text, candidates)
    if not matches or matches[0].start <= 0:
        return tuple(), tuple()

    leading_location = location_only_text(text[:matches[0].start])
    if leading_location is None:
        return tuple(), tuple()

    remainder = text[matches[0].start:]
    add_items = paragraph_to_items(remainder, candidates, existing_people)
    if not add_items:
        return tuple(), tuple()

    scratch_proposals = list(proposals)
    if attach_location_to_pending_item(
        scratch_proposals,
        block_id,
        leading_location,
        candidate_by_person,
    ):
        proposals[:] = scratch_proposals
        return add_items, tuple()

    update_items = existing_location_update(block, ns, leading_location, candidate_by_person, candidates)
    if not update_items:
        return tuple(), tuple()
    return add_items, update_items


def existing_location_update(
    block,
    ns: dict[str, str],
    raw_location: str,
    candidate_by_person: dict[str, NameCandidate],
    candidates: list[NameCandidate] | None = None,
) -> tuple[SignatureItem, ...]:
    loc_value = normalize(re.sub(r"^(i|från|fran)\s+", "", raw_location, flags=re.IGNORECASE))
    possible = []
    for item in block.findall(f".//{ns['tei_ns']}item[@type='signature']"):
        person_id = item.get("who")
        if not person_id:
            continue
        item_text = text_content(item)
        if person_id == "unknown":
            if candidates is None or not unknown_item_can_match_location(
                SignatureItem(
                    person_id="unknown",
                    text=item_text,
                    match_kind="existing-unknown-location-prefix",
                    repair_note="",
                    item_id=item.get(XML_ID, ""),
                ),
                raw_location,
                candidates,
            ):
                continue
        else:
            candidate = candidate_by_person.get(person_id)
            if candidate is None or loc_value not in candidate.locations:
                continue
        if loc_value in normalize(item_text):
            continue
        possible.append(SignatureItem(
            person_id=person_id,
            text=f"{item_text} {raw_location}",
            match_kind="leading-location-prefix",
            repair_note="",
            item_id=item.get(XML_ID, ""),
        ))
    if len(possible) != 1:
        return tuple()
    return tuple(possible)


def location_only_texts(text: str) -> tuple[str, ...]:
    if STOP_WORDS & set(tokenize(text)):
        return tuple()
    tokens = token_spans(text)
    if len(tokens) < 2:
        return tuple()
    if not is_location_marker_token(tokens[0].norm):
        return tuple()

    spans = []
    i = 0
    while i < len(tokens):
        if not is_location_marker_token(tokens[i].norm):
            return tuple()
        j = i + 1
        while j < len(tokens) and not is_location_marker_token(tokens[j].norm):
            if tokens[j].norm in PARTY_LABELS:
                return tuple()
            j += 1
        if j == i + 1:
            return tuple()
        spans.append(text[tokens[i].start:tokens[j - 1].end])
        i = j
    return tuple(spans)


def location_only_text(text: str) -> str | None:
    locations = location_only_texts(text)
    if len(locations) != 1:
        return None
    return locations[0]


def bare_location_only_text(text: str, candidates: list[NameCandidate]) -> str | None:
    if STOP_WORDS & set(tokenize(text)):
        return None
    if location_only_texts(text):
        return None
    tokens = token_spans(text)
    if not 1 <= len(tokens) <= 3:
        return None
    for token in tokens:
        if token.norm in PARTY_LABELS or is_location_marker_token(token.norm):
            return None
        if token.raw.isdigit() or len(token.norm) == 1:
            return None
        if not token.raw[:1].isupper():
            return None
    raw_location = text[tokens[0].start:tokens[-1].end]
    loc_value = normalize(raw_location)
    if loc_value not in candidate_location_values(candidates):
        return None
    return raw_location


def candidate_location_values(candidates: list[NameCandidate]) -> set[str]:
    values = set()
    for candidate in candidates:
        values.update(candidate.locations)
    return values


def location_text_is_known(raw_location: str, candidates: list[NameCandidate]) -> bool:
    loc_value = normalize(re.sub(r"^(i|från|fran)\s+", "", raw_location, flags=re.IGNORECASE))
    return loc_value in candidate_location_values(candidates)


def location_can_attach_to_item(
    item: SignatureItem,
    raw_location: str,
    candidate_by_person: dict[str, NameCandidate],
    candidates: list[NameCandidate],
) -> bool:
    loc_value = normalize(re.sub(r"^(i|från|fran)\s+", "", raw_location, flags=re.IGNORECASE))
    if loc_value in normalize(item.text):
        return False
    if item.person_id == "unknown":
        return unknown_item_can_match_location(item, raw_location, candidates)
    candidate = candidate_by_person.get(item.person_id)
    return candidate is not None and loc_value in candidate.locations


def unknown_item_can_match_location(
    item: SignatureItem,
    raw_location: str,
    candidates: list[NameCandidate],
) -> bool:
    if item.person_id != "unknown":
        return False
    loc_value = normalize(re.sub(r"^(i|från|fran)\s+", "", raw_location, flags=re.IGNORECASE))
    party_match = PARTY_LABEL_RE.search(item.text)
    if party_match is None:
        return False
    name_tokens = token_spans(item.text[:party_match.start()])
    if len(name_tokens) < 2:
        return False
    surname = name_tokens[-1].norm
    initial_tokens = name_tokens[:-1]
    for candidate in candidates:
        if loc_value not in candidate.locations:
            continue
        if not candidate.tokens or candidate.tokens[-1] != surname:
            continue
        if len(initial_tokens) > len(candidate.tokens) - 1:
            continue
        if all(candidate_token.startswith(token.norm) for token, candidate_token in zip(initial_tokens, candidate.tokens)):
            return True
    return False


def location_can_attach_to_previous_item(
    items: list[SignatureItem],
    raw_location: str,
    candidate_by_person: dict[str, NameCandidate],
    candidates: list[NameCandidate],
) -> bool:
    return location_attachment_item_index(items, raw_location, candidate_by_person, candidates) is not None


def location_attachment_item_index(
    items: list[SignatureItem],
    raw_location: str,
    candidate_by_person: dict[str, NameCandidate],
    candidates: list[NameCandidate],
) -> int | None:
    if not items:
        return None
    if location_can_attach_to_item(items[-1], raw_location, candidate_by_person, candidates):
        return len(items) - 1

    loc_value = normalize(re.sub(r"^(i|från|fran)\s+", "", raw_location, flags=re.IGNORECASE))
    possible = []
    for index, item in enumerate(items[:-1]):
        if loc_value in normalize(item.text):
            continue
        if unknown_item_can_match_location(item, raw_location, candidates):
            possible.append(index)
            continue
        if item.person_id == "unknown":
            continue
        candidate = candidate_by_person.get(item.person_id)
        if candidate is not None and loc_value in candidate.locations:
            possible.append(index)
    if len(possible) == 1:
        return possible[0]
    if possible or not fallback_single_match_location_is_safe(raw_location):
        return None
    if len(items) == 1:
        return 0
    return None



def split_leading_known_location(text: str, candidates: list[NameCandidate]) -> tuple[str, str] | None:
    tokens = token_spans(text)
    if len(tokens) < 4 or not is_location_marker_token(tokens[0].norm):
        return None

    locations = sorted(
        (tuple(tokenize(location)) for candidate in candidates for location in candidate.locations),
        key=len,
        reverse=True,
    )
    seen_locations = set()
    for location_tokens in locations:
        if not location_tokens or location_tokens in seen_locations:
            continue
        seen_locations.add(location_tokens)
        end_index = 1 + len(location_tokens)
        if end_index >= len(tokens):
            continue
        if tuple(token.norm for token in tokens[1:end_index]) != location_tokens:
            continue
        raw_location = text[tokens[0].start:tokens[end_index - 1].end]
        remainder = text[tokens[end_index].start:].strip()
        if remainder:
            return raw_location, remainder
    return None


def signature_name_prefix_is_safe(text: str) -> bool:
    text = re.sub(r"\s+", " ", text).strip(" ,;[]-=—–")
    if not text:
        return False
    tokens = token_spans(text)
    if not 2 <= len(tokens) <= 6:
        return False
    if STOP_WORDS & {token.norm for token in tokens}:
        return False
    name_like = 0
    initials = 0
    for token in tokens:
        if token.raw.isdigit():
            return False
        if len(token.norm) == 1:
            if not token.raw[:1].isupper():
                return False
            if not re.fullmatch(r"[A-Za-zÅÄÖåäö]", token.raw):
                return False
            initials += 1
            continue
        if not token.raw[:1].isupper():
            return False
        name_like += 1
    return name_like >= 2 or (name_like >= 1 and initials >= 1)


def strip_motion_label_prefix(text: str) -> tuple[str, str]:
    match = MOTION_LABEL_PREFIX_RE.match(text)
    if match is None:
        return text, ""
    stripped = text[match.end():].lstrip()
    party_match = PARTY_LABEL_RE.search(stripped)
    if party_match is None:
        return text, ""
    first_name = stripped[:party_match.start()]
    if not signature_name_prefix_is_safe(first_name):
        return text, ""
    return stripped, re.sub(r"\s+", " ", match.group(0)).strip()


def party_labelled_known_signature_item(
    raw_name: str,
    party: str,
    candidates: list[NameCandidate],
    seen_people: set[str],
) -> SignatureItem | None:
    tokens = token_spans(raw_name)
    if len(tokens) < 2:
        return None

    party_norm = normalize(party.strip("() "))
    possible = []
    for candidate in candidates:
        if not candidate_matches_observed_name_tokens(tokens, candidate):
            continue
        if candidate.parties and party_norm not in candidate.parties:
            continue
        possible.append(candidate)

    possible_people = {candidate.person_id for candidate in possible}
    if len(possible_people) != 1:
        return None
    person_id = next(iter(possible_people))
    if person_id in seen_people:
        return None
    return SignatureItem(
        person_id=person_id,
        text=f"{raw_name} {party}",
        match_kind="party-labelled-active-mp",
        repair_note="party-labelled-known",
    )


def paragraph_to_unknown_signature_items(
    text: str,
    candidates: list[NameCandidate],
    existing_people: set[str],
    block=None,
    ns: dict[str, str] | None = None,
    candidate_by_person: dict[str, NameCandidate] | None = None,
) -> tuple[tuple[SignatureItem, ...], tuple[SignatureItem, ...]]:
    if len(text) > 500:
        return tuple(), tuple()
    if STOP_WORDS & set(tokenize(text)):
        return tuple(), tuple()

    text, stripped_prefix = strip_motion_label_prefix(text)

    if re.search(r"\([^)]*\)", text) and any(
        not normalize(match.group(1)) in PARTY_LABELS
        for match in re.finditer(r"\(([^)]*)\)", text)
    ):
        return tuple(), tuple()

    party_matches = list(PARTY_LABEL_RE.finditer(text))
    if not party_matches:
        return tuple(), tuple()

    items = []
    update_items = tuple()
    first_leading_location = ""
    seen_people = set(existing_people)
    if candidate_by_person is None:
        candidate_by_person = {}
    for candidate in candidates:
        if candidate.person_id not in candidate_by_person:
            candidate_by_person[candidate.person_id] = candidate
    current_start = 0
    for index, match in enumerate(party_matches):
        raw_name = re.sub(r"\s+", " ", text[current_start:match.start()]).strip(" ,;[]|-—–")
        raw_location = ""
        leading = split_leading_known_location(raw_name, candidates)
        if leading is not None:
            leading_location, raw_name = leading
            if index == 0:
                first_leading_location = leading_location
            else:
                attachment_index = location_attachment_item_index(
                    items,
                    leading_location,
                    candidate_by_person,
                    candidates,
                )
                if attachment_index is None:
                    return tuple(), tuple()
                item = items[attachment_index]
                items[attachment_index] = replace(item, text=f"{item.text} {leading_location}")
        elif index == 0:
            pass
        elif is_location_marker_token(tokenize(raw_name)[0]) if tokenize(raw_name) else False:
            return tuple(), tuple()

        if re.search(r"[();]", raw_name):
            return tuple(), tuple()
        if not signature_name_prefix_is_safe(raw_name):
            return tuple(), tuple()

        party = re.sub(r"\s+", "", text[match.start():match.end()])
        item_text = f"{raw_name} {party}"
        if raw_location:
            item_text = f"{item_text} {raw_location}"
        known_item = party_labelled_known_signature_item(raw_name, party, candidates, seen_people)
        if known_item is not None:
            note = known_item.repair_note
            if stripped_prefix:
                note = f"{note};stripped-motion-label:{stripped_prefix}"
            known_item = replace(known_item, repair_note=note)
            items.append(known_item)
            seen_people.add(known_item.person_id)
        elif any(match.person_id in seen_people for match in find_name_matches(item_text, candidates)):
            return tuple(), tuple()
        else:
            repair_note = "party-labelled-unknown"
            if stripped_prefix:
                repair_note = f"{repair_note};stripped-motion-label:{stripped_prefix}"
            items.append(SignatureItem(
                person_id="unknown",
                text=item_text,
                match_kind="unknown-party-signature",
                repair_note=repair_note,
            ))
        current_start = match.end()

    suffix = re.sub(r"\s+", " ", text[current_start:]).strip(" ,;[]-=—–")
    if suffix:
        raw_location = location_only_text(suffix)
        if raw_location is None:
            return tuple(), tuple()
        attachment_index = location_attachment_item_index(
            items,
            raw_location,
            candidate_by_person,
            candidates,
        )
        if attachment_index is None:
            return tuple(), tuple()
        item = items[attachment_index]
        items[attachment_index] = replace(item, text=f"{item.text} {raw_location}")

    if not items:
        return tuple(), tuple()

    if first_leading_location:
        if block is not None and ns is not None:
            update_items = existing_location_update(
                block,
                ns,
                first_leading_location,
                candidate_by_person,
                candidates,
            )
        if not update_items:
            if not location_can_attach_to_item(items[0], first_leading_location, candidate_by_person, candidates):
                return tuple(), tuple()
            first = items[0]
            items[0] = replace(first, text=f"{first.text} {first_leading_location}")

    return tuple(items), update_items


def attach_locations_to_pending_items(
    proposals: list[Proposal],
    block_id: str,
    raw_locations: tuple[str, ...],
    candidate_by_person: dict[str, NameCandidate],
) -> bool:
    if not raw_locations:
        return False

    if len(raw_locations) > 1 and attach_locations_to_pending_suffix(
        proposals,
        block_id,
        raw_locations,
        candidate_by_person,
    ):
        return True

    updated: dict[int, list[SignatureItem]] = {}
    assigned_slots: set[tuple[int, int]] = set()
    for raw_location in raw_locations:
        loc_value = normalize(re.sub(r"^(i|från|fran)\s+", "", raw_location, flags=re.IGNORECASE))
        possible = []
        for proposal_index, proposal in enumerate(proposals):
            if proposal.signature_block_id != block_id:
                continue
            items = updated.get(proposal_index, list(proposal.add_items))
            for item_index, item in enumerate(items):
                if (proposal_index, item_index) in assigned_slots:
                    continue
                candidate = candidate_by_person.get(item.person_id)
                if candidate is None or loc_value not in candidate.locations:
                    continue
                if loc_value in normalize(item.text):
                    continue
                possible.append((proposal_index, item_index))
        if len(possible) != 1:
            return False

        proposal_index, item_index = possible[0]
        if proposal_index not in updated:
            updated[proposal_index] = list(proposals[proposal_index].add_items)
        item = updated[proposal_index][item_index]
        updated[proposal_index][item_index] = replace(item, text=f"{item.text} {raw_location}")
        assigned_slots.add((proposal_index, item_index))

    for proposal_index, items in updated.items():
        proposals[proposal_index] = replace(proposals[proposal_index], add_items=tuple(items))
    return True


def attach_locations_to_pending_suffix(
    proposals: list[Proposal],
    block_id: str,
    raw_locations: tuple[str, ...],
    candidate_by_person: dict[str, NameCandidate],
) -> bool:
    pending = []
    for proposal_index, proposal in enumerate(proposals):
        if proposal.signature_block_id != block_id:
            continue
        for item_index, item in enumerate(proposal.add_items):
            pending.append((proposal_index, item_index, item))
    if len(pending) < len(raw_locations):
        return False

    suffix = pending[-len(raw_locations):]
    loc_values = [
        normalize(re.sub(r"^(i|från|fran)\s+", "", raw_location, flags=re.IGNORECASE))
        for raw_location in raw_locations
    ]
    if len(set(loc_values)) != len(loc_values):
        return False
    for (_, _, item), loc_value in zip(suffix, loc_values):
        candidate = candidate_by_person.get(item.person_id)
        if candidate is None or loc_value not in candidate.locations:
            return False
        if loc_value in normalize(item.text):
            return False

    updated: dict[int, list[SignatureItem]] = {}
    for (proposal_index, item_index, item), raw_location in zip(suffix, raw_locations):
        if proposal_index not in updated:
            updated[proposal_index] = list(proposals[proposal_index].add_items)
        updated[proposal_index][item_index] = replace(item, text=f"{item.text} {raw_location}")
    for proposal_index, items in updated.items():
        proposals[proposal_index] = replace(proposals[proposal_index], add_items=tuple(items))
    return True


def attach_location_to_pending_item(
    proposals: list[Proposal],
    block_id: str,
    raw_location: str,
    candidate_by_person: dict[str, NameCandidate],
) -> bool:
    return attach_locations_to_pending_items(proposals, block_id, (raw_location,), candidate_by_person)


def attach_bare_location_to_pending_item(
    proposals: list[Proposal],
    block_id: str,
    raw_location: str,
    candidate_by_person: dict[str, NameCandidate],
) -> bool:
    loc_value = normalize(raw_location)
    possible = []
    for proposal_index, proposal in enumerate(proposals):
        if proposal.signature_block_id != block_id:
            continue
        for item_index, item in enumerate(proposal.add_items):
            if loc_value in normalize(item.text):
                continue
            candidate = candidate_by_person.get(item.person_id)
            if candidate is not None and loc_value in candidate.locations:
                possible.append((proposal_index, item_index))
    if len(possible) != 1:
        return False

    proposal_index, item_index = possible[0]
    items = list(proposals[proposal_index].add_items)
    item = items[item_index]
    items[item_index] = replace(item, text=f"{item.text} {raw_location}")
    proposals[proposal_index] = replace(proposals[proposal_index], add_items=tuple(items))
    return True


def find_or_create_signature_list(block, motion: str):
    for child in block:
        if child.tag.rsplit("}", 1)[-1] == "list":
            return child
    list_id = get_formatted_uuid(seed=f"{motion}:signature-list:{block.get(XML_ID, '')}")
    sig_list = etree.Element(f"{TEI_NS}list", attrib={XML_ID: list_id})
    block.append(sig_list)
    return sig_list


def iter_signature_blocks(root, ns: dict[str, str]) -> list:
    return root.findall(f".//{ns['tei_ns']}signatureBlock")


def signature_block_is_label_only(block, ns: dict[str, str]) -> bool:
    items = block.findall(f".//{ns['tei_ns']}item[@type='signature']")
    if not items:
        return False
    if any(item.get("who") and item.get("who") != "unknown" for item in items):
        return False

    text = text_content(block)
    if PARTY_LABEL_RE.search(text):
        return False
    if not LABEL_ONLY_SIGNATURE_RE.match(text):
        return False
    return bool(re.search(r"(?:mot|motion|\d{4}/\d{2})", text, flags=re.IGNORECASE))


def is_before_first_title_string(root, elem, ns: dict[str, str]) -> bool:
    first_title = root.find(f".//{ns['tei_ns']}p[@type='titleString']")
    if first_title is None:
        return False
    for current in root.iter():
        if current is elem:
            return True
        if current is first_title:
            return False
    return False


def iter_candidate_signature_blocks(root, ns: dict[str, str]) -> Iterable:
    for block in iter_signature_blocks(root, ns):
        if is_before_first_title_string(root, block, ns):
            continue
        if signature_block_is_label_only(block, ns):
            continue
        yield block


def iter_following_signature_tail(block, motion_number: int | None, limit: int = 12) -> Iterable:
    parent = block.getparent()
    if parent is None:
        return
    siblings = list(parent)
    block_index = siblings.index(block)
    for elem in siblings[block_index + 1:block_index + 1 + limit]:
        if is_boundary_or_body_text(elem, motion_number):
            break
        yield elem


def propose_for_motion(
    motion: str,
    person_index: PersonIndex,
    include_unknown_signatures: bool = False,
) -> list[Proposal]:
    metadata = infer_metadata(motion)
    year = int(metadata["year"])
    chamber = metadata.get("chamber", "")
    motion_number = metadata.get("number")
    parliament_year = Path(motion).parts[1]
    candidates = person_index.active_candidates(year, chamber)
    if not candidates:
        return []

    root, ns = parse_tei(motion)
    proposals = []
    candidate_by_person = {}
    for candidate in candidates:
        if candidate.person_id not in candidate_by_person:
            candidate_by_person[candidate.person_id] = candidate
    for block in iter_candidate_signature_blocks(root, ns):
        sig_list = find_or_create_signature_list(block, motion)
        existing_people = existing_signature_people(block, ns)
        block_id = block.get(XML_ID, "")

        for elem in iter_following_signature_tail(block, motion_number):
            paragraph_text = text_content(elem)
            add_items = paragraph_to_items(paragraph_text, candidates, existing_people)
            update_items = tuple()
            if not add_items:
                add_items, update_items = paragraph_to_items_after_leading_location(
                    paragraph_text,
                    candidates,
                    existing_people,
                    proposals,
                    block,
                    ns,
                    block_id,
                    candidate_by_person,
                )
            if not add_items and include_unknown_signatures:
                add_items, update_items = paragraph_to_unknown_signature_items(
                    paragraph_text,
                    candidates,
                    existing_people,
                    block,
                    ns,
                    candidate_by_person,
                )
            if not add_items:
                if has_pending_signature_items(proposals, block_id):
                    add_items = paragraph_to_abbreviated_continuation_items(
                        paragraph_text,
                        candidates,
                        existing_people,
                        proposals,
                        block_id,
                        candidate_by_person,
                    )
                if add_items:
                    proposals.append(Proposal(
                        motion=motion,
                        parliament_year=parliament_year,
                        calendar_year=year,
                        chamber=chamber,
                        signature_block_id=block_id,
                        list_id=sig_list.get(XML_ID, ""),
                        paragraph_id=elem.get(XML_ID, ""),
                        paragraph_text=paragraph_text,
                        add_items=add_items,
                        update_items=update_items,
                    ))
                    existing_people.update(item.person_id for item in add_items)
                    continue

                if has_pending_signature_items(proposals, block_id):
                    bare_location = bare_location_only_text(paragraph_text, candidates)
                    if bare_location and attach_bare_location_to_pending_item(
                        proposals,
                        block_id,
                        bare_location,
                        candidate_by_person,
                    ):
                        proposals.append(Proposal(
                            motion=motion,
                            parliament_year=parliament_year,
                            calendar_year=year,
                            chamber=chamber,
                            signature_block_id=block_id,
                            list_id=sig_list.get(XML_ID, ""),
                            paragraph_id=elem.get(XML_ID, ""),
                            paragraph_text=paragraph_text,
                            add_items=tuple(),
                        ))
                        continue

                    add_items = paragraph_to_single_name_continuation_item(
                        paragraph_text,
                        candidates,
                        existing_people,
                    )
                if add_items:
                    proposals.append(Proposal(
                        motion=motion,
                        parliament_year=parliament_year,
                        calendar_year=year,
                        chamber=chamber,
                        signature_block_id=block_id,
                        list_id=sig_list.get(XML_ID, ""),
                        paragraph_id=elem.get(XML_ID, ""),
                        paragraph_text=paragraph_text,
                        add_items=add_items,
                        update_items=update_items,
                    ))
                    existing_people.update(item.person_id for item in add_items)
                    continue

                raw_locations = location_only_texts(paragraph_text)
                if not attach_locations_to_pending_items(
                    proposals,
                    block_id,
                    raw_locations,
                    candidate_by_person,
                ):
                    break
                proposals.append(Proposal(
                    motion=motion,
                    parliament_year=parliament_year,
                    calendar_year=year,
                    chamber=chamber,
                    signature_block_id=block_id,
                    list_id=sig_list.get(XML_ID, ""),
                    paragraph_id=elem.get(XML_ID, ""),
                    paragraph_text=paragraph_text,
                    add_items=tuple(),
                ))
                continue
            proposals.append(Proposal(
                motion=motion,
                parliament_year=parliament_year,
                calendar_year=year,
                chamber=chamber,
                signature_block_id=block_id,
                list_id=sig_list.get(XML_ID, ""),
                paragraph_id=elem.get(XML_ID, ""),
                paragraph_text=paragraph_text,
                add_items=add_items,
                update_items=update_items,
            ))
            existing_people.update(item.person_id for item in add_items)
    return proposals


def apply_proposals_to_motion(motion: str, proposals: list[Proposal]) -> int:
    if not proposals:
        return 0
    root, ns = parse_tei(motion)
    motion_number = infer_metadata(motion).get("number")
    by_block: dict[str, dict[str, Proposal]] = defaultdict(dict)
    for proposal in proposals:
        if proposal.signature_block_id:
            by_block[proposal.signature_block_id][proposal.paragraph_id] = proposal
    applied = 0
    for block in iter_candidate_signature_blocks(root, ns):
        target_proposals = by_block.get(block.get(XML_ID, ""))
        if not target_proposals:
            continue
        sig_list = find_or_create_signature_list(block, motion)
        parent = block.getparent()
        if parent is None:
            continue
        for elem in list(iter_following_signature_tail(block, motion_number)):
            paragraph_id = elem.get(XML_ID)
            proposal = target_proposals.get(paragraph_id)
            if proposal is None:
                break
            update_failed = False
            for update_item in proposal.update_items:
                if update_item.item_id:
                    matches = [
                        item
                        for item in block.findall(f".//{ns['tei_ns']}item[@type='signature']")
                        if item.get(XML_ID) == update_item.item_id
                    ]
                else:
                    matches = [
                        item
                        for item in block.findall(f".//{ns['tei_ns']}item[@type='signature']")
                        if item.get("who") == update_item.person_id
                    ]
                if len(matches) != 1:
                    update_failed = True
                    break
                matches[0].text = update_item.text
            if update_failed:
                break
            for item_index, signature_item in enumerate(proposal.add_items, start=1):
                item_id = get_formatted_uuid(
                    seed=f"{motion}:{paragraph_id}:{item_index}:{signature_item.person_id}:{signature_item.text}"
                )
                item = etree.Element(
                    f"{TEI_NS}item",
                    attrib={XML_ID: item_id, "who": signature_item.person_id, "type": "signature"},
                )
                item.text = signature_item.text
                sig_list.append(item)
            parent.remove(elem)
            applied += 1
    if applied:
        write_tei(root, motion)
    return applied


def proposal_rows(proposals: Iterable[Proposal]) -> list[dict[str, str | int]]:
    rows = []
    for proposal in proposals:
        rows.append({
            "motion": proposal.motion,
            "parliament_year": proposal.parliament_year,
            "calendar_year": proposal.calendar_year,
            "chamber": proposal.chamber,
            "signature_block_id": proposal.signature_block_id,
            "list_id": proposal.list_id,
            "paragraph_id": proposal.paragraph_id,
            "paragraph_text": proposal.paragraph_text,
            "n_items": len(proposal.add_items),
            "add_people": " | ".join(item.person_id for item in proposal.add_items),
            "add_text": " | ".join(item.text for item in proposal.add_items),
            "match_kinds": " | ".join(item.match_kind for item in proposal.add_items),
            "repair_notes": " | ".join(item.repair_note for item in proposal.add_items),
            "n_updates": len(proposal.update_items),
            "update_people": " | ".join(item.person_id for item in proposal.update_items),
            "update_text": " | ".join(item.text for item in proposal.update_items),
            "n_unknown": sum(1 for item in proposal.add_items if item.person_id == "unknown"),
            "unknown_text": " | ".join(item.text for item in proposal.add_items if item.person_id == "unknown"),
            "unknown_reasons": " | ".join(item.repair_note for item in proposal.add_items if item.person_id == "unknown"),
        })
    return rows


def write_tsv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "motion",
        "parliament_year",
        "calendar_year",
        "chamber",
        "signature_block_id",
        "list_id",
        "paragraph_id",
        "paragraph_text",
        "n_items",
        "add_people",
        "add_text",
        "match_kinds",
        "repair_notes",
        "n_updates",
        "update_people",
        "update_text",
        "n_unknown",
        "unknown_text",
        "unknown_reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str | int]], applied: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_year = defaultdict(lambda: {"paragraphs": 0, "items": 0})
    for row in rows:
        by_year[row["parliament_year"]]["paragraphs"] += 1
        by_year[row["parliament_year"]]["items"] += int(row["n_items"])
    repair_rows = [row for row in rows if "ocr-repair" in str(row.get("match_kinds", ""))]
    repair_items = sum(
        kind == "ocr-repair"
        for row in rows
        for kind in str(row.get("match_kinds", "")).split(" | ")
    )

    lines = [
        "# High-confidence signature block extension proposals",
        "",
        f"- Mode: `{'applied' if applied else 'dry-run'}`",
        f"- Paragraphs proposed: `{len(rows)}`",
        f"- Signature items proposed: `{sum(int(row['n_items']) for row in rows)}`",
        f"- Paragraphs with OCR-repaired names: `{len(repair_rows)}`",
        f"- OCR-repaired signature items: `{repair_items}`",
        "",
        "The script only proposes immediate following paragraphs that can be segmented into active MPs for the motion year/chamber.",
        "",
        "## Top years",
        "",
        "| parliamentary year | paragraphs | items |",
        "| --- | ---: | ---: |",
    ]
    for year, counts in sorted(by_year.items(), key=lambda item: (-item[1]["paragraphs"], str(item[0])))[:40]:
        lines.append(f"| {year} | {counts['paragraphs']} | {counts['items']} |")

    lines.extend(["", "## Examples", ""])
    for index, row in enumerate(rows[:60], start=1):
        lines.extend([
            f"### {index}. `{row['motion']}`",
            "",
            f"- Paragraph: `{row['paragraph_id']}`",
            f"- Add items: `{row['n_items']}`",
            f"- Add text: {row['add_text']}",
            f"- Source paragraph: {row['paragraph_text']}",
            "",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = fetch_parser("motions", docstring=__doc__)
    parser.add_argument("--persons-root", default="../riksdagen-persons")
    parser.add_argument("--out", default="test/results/signature-block-extension-high-confidence-proposals.tsv")
    parser.add_argument("--markdown-out", default="docs/issue-drafts/signature-block-extension-high-confidence-proposals.md")
    parser.add_argument("--apply", action="store_true", help="Apply proposed XML edits. Default is dry-run.")
    parser.add_argument(
        "--include-unknown-signatures",
        action="store_true",
        help="Also propose clearly party-labelled signature tail items with who='unknown'.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Analyze only the first N selected motions.")
    args = impute_args(parser.parse_args())

    motions = list(args.motions)
    if args.limit is not None:
        motions = motions[:args.limit]
    LOGGER.info(f"Analyzing {len(motions)} motions")

    person_index = PersonIndex(Path(args.persons_root))
    proposals = []
    for i, motion in enumerate(motions, start=1):
        if i % 5000 == 0:
            LOGGER.info(f"Processed {i} motions; proposals so far: {len(proposals)}")
        proposals.extend(propose_for_motion(
            motion,
            person_index,
            include_unknown_signatures=args.include_unknown_signatures,
        ))

    rows = proposal_rows(proposals)
    write_tsv(Path(args.out), rows)
    write_markdown(Path(args.markdown_out), rows, applied=args.apply)
    LOGGER.info(f"Wrote {args.out} and {args.markdown_out}")

    if args.apply:
        proposals_by_motion = defaultdict(list)
        for proposal in proposals:
            proposals_by_motion[proposal.motion].append(proposal)
        changed = 0
        paragraphs = 0
        for motion, motion_proposals in proposals_by_motion.items():
            applied = apply_proposals_to_motion(motion, motion_proposals)
            if applied:
                changed += 1
                paragraphs += applied
        LOGGER.info(f"Applied {paragraphs} paragraph moves across {changed} motions")


if __name__ == "__main__":
    main()
