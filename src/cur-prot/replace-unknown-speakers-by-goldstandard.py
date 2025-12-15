#!/usr/bin/env python3
"""
Scans speaker-segments folder (is-speaker / non-speaker) and applies the mappings.
Handles nested <u> and <seg> elements, propagates 'who' and 'type'.
Correctly propagates speaker along <u> next/prev chains and following siblings,
but only annotates <u> and <seg> blocks where who="unknown".
Counters now correctly sum using multiprocessing.
"""
import argparse
from multiprocessing import Pool, cpu_count
import os
import pandas as pd
from pyriksdagen.io import parse_tei, write_tei
from trainerlog import get_logger
from tqdm import tqdm

logger = None


def propagate_speaker_from_note(note_el, person_id):
    """
    Propagate speaker from a <note type="speaker"> to following <u> elements
    with who missing or 'unknown'. Never reads or writes @who on <note>.
    """
    modified = False

    parent = note_el.getparent()
    if parent is None:
        return modified

    # Lookup for chained <u> elements
    u_map = {
        u.get('xml:id'): u
        for u in parent.iter()
        if u.tag.endswith('u') and u.get('xml:id')
    }

    siblings = list(parent)
    try:
        idx = siblings.index(note_el)
    except ValueError:
        return modified

    for sib in siblings[idx + 1:]:
        # Stop at next speaker note
        if sib.tag.endswith('note') and sib.get('type') == 'speaker':
            break

        # Skip neutral elements
        if sib.tag.endswith(('note', 'pb')):
            continue

        if sib.tag.endswith('u'):
            if sib.get('who') in (None, 'unknown'):
                sib.set('who', person_id)
                modified = True

            # Nested <u>
            for child in sib.iter():
                if child.tag.endswith('u') and child.get('who') in (None, 'unknown'):
                    child.set('who', person_id)
                    modified = True

            # Follow @next chain
            next_id = sib.get('next')
            while next_id:
                next_el = u_map.get(next_id)
                if next_el is None:
                    break
                if next_el.get('who') in (None, 'unknown'):
                    next_el.set('who', person_id)
                    modified = True
                next_id = next_el.get('next')

    return modified


def apply_speaker_on_note_sibling(note_el, person_id):
    """
    Find the first following <u> with who missing/unknown and propagate speaker.
    """
    parent = note_el.getparent()
    if parent is None:
        return False

    found_note = False
    for el in parent:
        if el == note_el:
            found_note = True
            continue
        if found_note and el.tag.endswith('u') and el.get('who') in (None, 'unknown'):
            return propagate_speaker_from_note(note_el, person_id)

    return False


def apply_speaker_recursively(el, person_id, folder_type):
    modified = False

    if folder_type == 'is-speaker':
        if el.tag.endswith('note'):
            if el.get('type') != 'speaker':
                el.set('type', 'speaker')
                modified = True

            propagated = apply_speaker_on_note_sibling(el, person_id)
            if propagated:
                modified = True


        elif el.tag.endswith('u'):
            if el.get('who') in (None, 'unknown'):
                el.set('who', person_id)
                modified = True

    else:  # non-speaker
        if el.tag.endswith(('u', 'seg')) and el.get('who') == 'unknown':
            el.attrib.pop('who', None)
            modified = True
        if el.tag.endswith('note') and el.get('type') == 'speaker':
            el.attrib.pop('type', None)
            modified = True

    for child in el:
        modified |= apply_speaker_recursively(child, person_id, folder_type)

    return modified


def handle_row_on_element(el, row, folder_type):
    person_id = row.get('person_id')
    modified = apply_speaker_recursively(el, person_id, folder_type)
    if modified:
        if folder_type == 'is-speaker':
            message = 'speaker added/updated'
        else:
            message = 'non-speaker cleaned'
        return 'success', message
    else:
        return 'already_fixed', 'already correct'


def group_rows_by_folder(input_path, logger):
    """
    Scan 'is-speaker' and 'non-speaker' subfolders, read TSV files, and
    group rows by protocol_id. Logs a warning if a folder is missing.
    """
    groups = {}

    for folder in ['is-speaker', 'non-speaker']:
        folder_path = os.path.join(input_path, folder)
        if not os.path.exists(folder_path):
            logger.warning(f"Folder not found, skipping: {folder_path}")
            continue

        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith('.tsv'):
                continue
            csv_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(csv_path, sep="\t", dtype=str).fillna('')
            for idx, row in df.iterrows():
                if 'protocol_id' not in row or not row['protocol_id']:
                    logger.error(f"Row {idx} in {csv_path} has no protocol_id, skipping")
                    continue

                xml_path = os.path.join("riksdagen-records", row['protocol_id'])
                uuid = row.get('uuid')
                if not uuid:
                    logger.error(f"Row {idx} in {csv_path} has no UUID, skipping")
                    continue

                if xml_path not in groups:
                    groups[xml_path] = []
                groups[xml_path].append({
                    'index': idx,
                    'uuid': uuid,
                    **row.to_dict(),
                    'folder_type': folder
                })
                
    return groups


def find_element_by_xml_id(root, uuid):
    """
    Find an element by its xml:id using XPath and proper namespace handling.
    Returns None if not found.
    """
    if not uuid:
        return None

    ns = {
        'xml': 'http://www.w3.org/XML/1998/namespace',
        'tei': 'http://www.tei-c.org/ns/1.0'
    }

    xpath_expr = f".//*[@xml:id='{uuid}']"
    result = root.xpath(xpath_expr, namespaces=ns)

    return result[0] if result else None


def process_file_task(file_path, rows):
    """
    Process a single TEI file and apply speaker/non-speaker updates.
    Uses pyriksdagen.io.write_tei() to serialize changes back to disk.

    Returns:
        tuple: (num_success, num_already_fixed, num_failures, failure_list)
    """
    success_rows, already_fixed_rows, failures = [], [], []

    if not os.path.exists(file_path):
        failures.extend([(r['index'], file_path, "File not found") for r in rows])
        return 0, 0, len(failures), failures

    try:
        root, ns = parse_tei(file_path, get_ns=True)
    except Exception as e:
        failures.extend([(r['index'], file_path, f"Failed to parse XML: {e}") for r in rows])
        return 0, 0, len(failures), failures

    for r in rows:
        idx = r['index']
        folder_type = r['folder_type']
        uuid = r.get('uuid')

        if not uuid:
            failures.append((idx, file_path, "No UUID provided in row"))
            continue

        el = find_element_by_xml_id(root, uuid)
        if el is None:
            failures.append((idx, file_path, f"Element with xml:id={uuid} not found"))
            continue

        result, _ = handle_row_on_element(el, r, folder_type)
        if result == 'success':
            success_rows.append(idx)
        else:
            already_fixed_rows.append(idx)

    if success_rows:
        try:
            write_tei(root, file_path)
        except Exception as e:
            failures.extend([(idx, file_path, f"Failed to write back XML: {e}") for idx in success_rows])
            success_rows.clear()

    return len(success_rows), len(already_fixed_rows), len(failures), failures



def process_file_task_star(args):
    """Unpack tuple for multiprocessing"""
    return process_file_task(*args)

def add_row_to_grouped(groups, row, folder_type, logger=None):
    """
    Add a row to the grouped dictionary by protocol_id.
    Constructs the full XML path and ensures UUID exists.
    Returns True if added successfully, False otherwise.
    """
    idx = row.get('index')
    protocol_id = row.get('protocol_id')
    if not protocol_id:
        if logger:
            logger.error(f"Row {idx} has no protocol_id, skipping")
        return False

    xml_path = os.path.join("riksdagen-records", protocol_id)
    uuid = row.get('uuid')
    if not uuid:
        if logger:
            logger.error(f"Row {idx} in {xml_path} has no UUID, skipping")
        return False

    if xml_path not in groups:
        groups[xml_path] = []

    groups[xml_path].append({
        'index': idx,
        'uuid': uuid,
        **row.to_dict(),
        'folder_type': folder_type
    })
    return True

def main(args):
    logger = get_logger(name="speaker_mapper", level=args.loglevel)
    input_path = args.folder
    grouped = {}

    if os.path.isfile(input_path) and input_path.lower().endswith(('.tsv', '.csv')):
        logger.debug(f"Reading single file: {input_path}")
        df = pd.read_csv(
            input_path, 
            sep="\t" if input_path.endswith('.tsv') else ",", 
            dtype=str
        ).fillna('')
        for idx, row in df.iterrows():
            row['index'] = idx 
            add_row_to_grouped(grouped, row, folder_type='is-speaker', logger=logger)

    elif os.path.isdir(input_path):
        logger.debug(f"Scanning directory: {input_path}")
        grouped = group_rows_by_folder(input_path, logger)
    else:
        logger.error(f"Path not found or not valid: {input_path}")
        return

    tasks = [(file_path, rows) for file_path, rows in grouped.items()]
    if not tasks:
        logger.info("No files to process.")
        return

    total_rows = sum(len(rows) for rows in grouped.values())
    n_workers = min(cpu_count() or 1, max(1, len(tasks)))
    logger.info(f"Processing {len(tasks)} files using {n_workers} workers...")

    total_success = 0
    total_already_fixed = 0
    total_failures = 0
    all_failures_list = []

    if args.multithread:
        logger.info(f"Processing {len(tasks)} files using {n_workers} workers (multithreaded)...")
        with Pool(n_workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_file_task_star, tasks), total=len(tasks)))
    else:
        logger.info(f"Processing {len(tasks)} files using a single thread...")
        results = [process_file_task(file_path, rows) for file_path, rows in tqdm(tasks)]

    for succ_count, fixed_count, fail_count, fail_list in results:
        total_success += succ_count
        total_already_fixed += fixed_count
        total_failures += fail_count
        all_failures_list.extend([(i, fpath, reason) for (i, fpath, reason) in fail_list])

    if all_failures_list:
        fail_df = pd.DataFrame(all_failures_list, columns=['row_index', 'file_path', 'reason'])
        fail_df.to_csv("input/matching/speaker-mapping-failures.tsv", sep="\t", index=False)
        logger.warning(f"Written {len(all_failures_list)} failures to input/matching/speaker_mapping_failures.tsv")

    logger.info(f"\nSummary:")
    logger.info(f"Total rows scanned       : {total_rows}")
    logger.info(f"Successful modifications : {total_success}")
    logger.info(f"Already fixed            : {total_already_fixed}")
    logger.info(f"Failures reported        : {total_failures}")

    if total_success + total_already_fixed + total_failures != total_rows:
        logger.warning("Totals do not match total rows scanned!")

    if total_failures:
        logger.error("Some rows were not applied correctly or failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder",
        required=True,
        help="Base folder containing is-speaker / non-speaker subfolders, or a TSV/CSV file with failures."
    )
    parser.add_argument(
        "--multithread",
        action="store_true",
        help="Enable multithreading for processing files (default: False)."
    )
    parser.add_argument(
        "--loglevel",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: DEBUG)"
    )
    args = parser.parse_args()

    main(args)