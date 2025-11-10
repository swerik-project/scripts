#!/usr/bin/env python3
"""
replace_unknown_speakers_goldstandard.py

Scans speaker-segments folder (is-speaker / non-speaker) and applies the mappings.
Handles nested <u> and <seg> elements, propagates 'who' and 'type'.
Correctly propagates speaker along <u> next/prev chains and following siblings,
but only annotates <u> and <seg> blocks where who="unknown".
Counters now correctly sum using multiprocessing.
"""
import argparse
from collections import defaultdict
from lxml import etree
from multiprocessing import Pool, cpu_count
import os
import pandas as pd
from pyriksdagen.io import parse_tei, write_tei
import sys
from typing import Dict, Tuple

def propagate_speaker_from_note(note_el: etree._Element):
    """
    After a <note type="speaker" who="...">, propagate the speaker ID
    to all following <u> siblings (and their next chain) with who="unknown",
    even if <note> or <pb> elements appear in between.
    Stops when another <note type="speaker"> appears.
    """
    modified = False
    speaker_id = note_el.get('who')
    if not speaker_id:
        return modified

    parent = note_el.getparent()
    if parent is None:
        return modified

    # Build a lookup map for <u> by xml:id
    u_map = {u.get('xml:id'): u for u in parent.iter() if u.tag.endswith('u') and u.get('xml:id')}

    siblings = list(parent)
    try:
        idx = siblings.index(note_el)
    except ValueError:
        return modified

    for sib in siblings[idx + 1:]:
        # Stop if a new speaker note appears
        if sib.tag.endswith('note') and sib.get('type') == 'speaker':
            break

        # Skip over neutral notes or page breaks, but continue later
        if sib.tag.endswith(('note', 'pb')):
            continue

        if sib.tag.endswith('u'):
            # Update <u> if who unknown or missing
            if sib.get('who') in (None, 'unknown'):
                sib.set('who', speaker_id)
                modified = True

            # Update any nested <u>
            for child in sib.iter():
                if child.tag.endswith(('u')) and child.get('who') in (None, 'unknown'):
                    child.set('who', speaker_id)
                    modified = True

            # Follow next chain even if other siblings appear between
            next_id = sib.get('next')
            while next_id:
                next_el = u_map.get(next_id)
                if next_el is None:
                    break
                if next_el.get('who') in (None, 'unknown'):
                    next_el.set('who', speaker_id)
                    modified = True
                for child in next_el.iter():
                    if child.tag.endswith(('u')) and child.get('who') in (None, 'unknown'):
                        child.set('who', speaker_id)
                        modified = True
                next_id = next_el.get('next')

    return modified


def apply_speaker_on_note_sibling(note_el: etree._Element):
    """Annotate the first <u> sibling after note with who='unknown', then propagate."""
    modified = False
    parent = note_el.getparent()
    if parent is None:
        return modified

    found_note = False
    for el in parent:
        if el == note_el:
            found_note = True
            continue
        if found_note and el.tag.endswith('u') and (el.get('who') in (None, 'unknown')):
            propagate_speaker_from_note(note_el)
            modified = True
            break

    return modified


def apply_speaker_recursively(el: etree._Element, person_id: str, folder_type: str) -> bool:
    """Recursively update 'who' and 'type' attributes for speaker/non-speaker blocks."""
    modified = False

    if folder_type == 'is-speaker':
        if el.tag.endswith('note'):
            if el.get('type') != 'speaker':
                el.set('type', 'speaker')
                modified = True
            if el.get('who') != person_id:
                el.set('who', person_id)
                modified = True
            # Propagate to next <u> chain if applicable
            if apply_speaker_on_note_sibling(el):
                modified = True
        elif el.tag.endswith('u'):
            if el.get('who') in (None, 'unknown'):
                el.set('who', person_id)
                modified = True
    else:  # non-speaker
        if el.tag.endswith(('note', 'u', 'seg')) and el.get('who') == 'unknown':
            el.attrib.pop('who', None)
            modified = True
        if el.tag.endswith('note') and el.get('type') == 'speaker':
            el.attrib.pop('type', None)
            modified = True

    for child in el:
        if apply_speaker_recursively(child, person_id, folder_type):
            modified = True

    return modified


def handle_row_on_element(el: etree._Element, row: Dict, folder_type: str) -> Tuple[str, str]:
    person_id = row.get('person_id')
    modified = apply_speaker_recursively(el, person_id, folder_type)
    if modified:
        return 'success', 'speaker added/updated' if folder_type == 'is-speaker' else 'non-speaker cleaned'
    else:
        return 'already_fixed', 'already correct'


def group_rows_by_folder(data_base: str) -> Dict[str, list]:
    groups = defaultdict(list)
    for folder in ['is-speaker', 'non-speaker']:
        folder_path = os.path.join(data_base, folder)
        if not os.path.exists(folder_path):
            continue
        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith(".tsv"):
                continue
            csv_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(csv_path, sep="\t", dtype=str).fillna('')
            for idx, row in df.iterrows():
                xml_path = row['protocol_id']
                groups[xml_path].append({'index': idx, **row.to_dict(), 'folder_type': folder})
    return groups


def find_element_by_xml_id(root: etree._Element, uuid: str) -> etree._Element:
    xml_id_attr = '{http://www.w3.org/XML/1998/namespace}id'
    for el in root.iter():
        if el.get(xml_id_attr) == uuid:
            return el
    return None


def process_file_task(args):
    file_path, rows = args
    success_rows = []
    already_fixed_rows = []
    failures = []

    if not os.path.exists(file_path):
        for r in rows:
            failures.append((r['index'], f"File not found: {file_path}"))
        return len(success_rows), len(already_fixed_rows), len(failures), failures

    try:
        root, ns = parse_tei(file_path, get_ns=True)
    except Exception as e:
        for r in rows:
            failures.append((r['index'], f"Failed to parse XML: {e}"))
        return len(success_rows), len(already_fixed_rows), len(failures), failures

    for r in rows:
        idx = r['index']
        folder_type = r['folder_type']
        uuid = r.get('uuid')
        if not uuid:
            failures.append((idx, f"No UUID provided in row"))
            continue
        el = find_element_by_xml_id(root, uuid)
        if el is None:
            failures.append((idx, f"Element with xml:id={uuid} not found"))
            continue


        result, msg = handle_row_on_element(el, r, folder_type)
        if result == 'success':
            success_rows.append(idx)
        else:
            already_fixed_rows.append(idx)

    if success_rows:
        try:
            write_tei(root, file_path)
        except Exception as e:
            for idx in success_rows:
                failures.append((idx, f"Failed to write back file: {e}"))
            success_rows = []

    return len(success_rows), len(already_fixed_rows), len(failures), failures


def main(args):
    input_path = args.folder

    grouped = defaultdict(list)

    # If it's a single TSV/CSV file
    if os.path.isfile(input_path) and input_path.lower().endswith(('.tsv', '.csv')):
        df = pd.read_csv(input_path, sep="\t" if input_path.endswith('.tsv') else ",", dtype=str).fillna('')
        # Group by protocol_id
        for idx, row in df.iterrows():
            xml_path = row['protocol_id']
            # pick the correct element uuid
            uuid = row.get('new_uuid') or row.get('original_uuid')
            grouped[xml_path].append({
                'index': idx,
                'uuid': uuid,
                **row.to_dict(),
                'folder_type': 'is-speaker'  # assume is-speaker failures
            })
    elif os.path.isdir(input_path):
        # Scan the folder for is-speaker / non-speaker
        grouped = group_rows_by_folder(input_path)
    else:
        print(f"[ERROR] Path not found or not valid: {input_path}")
        sys.exit(1)

    tasks = [(file_path, rows) for file_path, rows in grouped.items()]

    if not tasks:
        print("[INFO] No files to process.")
        sys.exit(0)

    total_rows = sum(len(rows) for rows in grouped.values())
    n_workers = min(cpu_count() or 1, max(1, len(tasks)))
    print(f"Processing {len(tasks)} files using {n_workers} workers...")

    total_success = 0
    total_already_fixed = 0
    total_failures = 0
    all_failures_list = []

    with Pool(n_workers) as pool:
        results = pool.map(process_file_task, tasks)

    for succ_count, fixed_count, fail_count, fail_list in results:
        total_success += succ_count
        total_already_fixed += fixed_count
        total_failures += fail_count
        all_failures_list.extend([(i, reason) for (i, reason) in fail_list])

    if all_failures_list:
        fail_df = pd.DataFrame(all_failures_list, columns=['row_index', 'reason'])
        fail_df.to_csv("speaker_mapping_failures.tsv", sep="\t", index=False)
        print(f"Written {len(all_failures_list)} failures to speaker_mapping_failures.tsv")

    print(f"\nSummary:")
    print(f"  Total rows scanned       : {total_rows}")
    print(f"  Successful modifications : {total_success}")
    print(f"  Already fixed            : {total_already_fixed}")
    print(f"  Failures reported        : {total_failures}")

    if total_success + total_already_fixed + total_failures != total_rows:
        print("Warning: totals do not match total rows scanned!")

    if total_failures:
        print("Some rows were not applied correctly or failed.")
        sys.exit(3)
    else:
        print("All rows successfully applied.")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply speaker mappings from speaker-segments folders or single TSV/CSV files.")
    parser.add_argument("--folder", required=True, help="Base folder containing is-speaker / non-speaker subfolders, or a TSV/CSV file with failures.")
    args = parser.parse_args()
    main()
