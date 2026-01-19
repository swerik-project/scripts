"""
Add a randomly generated UUID to all elements in the XML ID attribute that are currently missing one.

Also adds the document ID (eg. prot-year--number) in the TEI element as an XML ID attribute if its missing.
"""
from collections import Counter
import multiprocessing
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from pyriksdagen.io import (
    parse_tei,
    write_tei
)
from pyriksdagen.utils import (
    elem_iter,
    get_formatted_uuid
)
from pyriksdagen.utils import (
        TEI_NS,
        XML_NS
)
from trainerlog import get_logger
from tqdm import tqdm

logger = get_logger(name="trainlog.add_protocol_id", level="DEBUG")

def add_protocol_id(protocol):
    ids = set()
    seed_map = {}
    root, ns = parse_tei(protocol)

    tei = root.find(f"{TEI_NS}TEI")
    if root.tag.split("}")[-1] == "TEI":
        tei = root

    doc_id = protocol.split("/")[-1][:-4]
    if f"{XML_NS}id" not in tei.attrib:
        tei.attrib[f"{XML_NS}id"] = doc_id

    num_ids = 0
    note_counter = 0
    u_counter = 0
    for tag, elem in elem_iter(root):
        if tag == "u":
            for idx, subelem in enumerate(elem):
                if f'{XML_NS}id' not in subelem.attrib:
                    seed_str = f"{doc_id}\n{tag}_sub\n{idx}"
                    id_val = get_formatted_uuid(seed_str)
                    subelem.attrib[f'{XML_NS}id'] = id_val
                    seed_map[id_val] = seed_str
                ids.add(subelem.attrib[f'{XML_NS}id'])
                num_ids += 1

            if f'{XML_NS}id' not in elem.attrib:
                seed_str = f"{doc_id}\n{tag}\n{u_counter}"
                id_val = get_formatted_uuid(seed_str)
                elem.attrib[f'{XML_NS}id'] = id_val
                seed_map[id_val] = seed_str
            ids.add(elem.attrib[f'{XML_NS}id'])
            num_ids += 1
            u_counter += 1

        elif tag == "note":
            if f'{XML_NS}id' not in elem.attrib:
                seed_str = f"{doc_id}\n{tag}\n{note_counter}"
                id_val = get_formatted_uuid(seed_str)
                elem.attrib[f'{XML_NS}id'] = id_val
                seed_map[id_val] = seed_str
            ids.add(elem.attrib[f'{XML_NS}id'])
            num_ids += 1
            note_counter += 1

    for body in root.findall(f".//{TEI_NS}body"):
        div_counter = 0
        for div in body:
            if f'{XML_NS}id' not in div.attrib:
                seed_str = f"{doc_id}\ndiv\n{div_counter}"
                id_val = get_formatted_uuid(seed_str)
                div.attrib[f'{XML_NS}id'] = id_val
                seed_map[id_val] = seed_str
            ids.add(div.attrib[f'{XML_NS}id'])
            num_ids += 1
            div_counter += 1

    write_tei(root, protocol)

    return ids, num_ids, seed_map


def main(args):
    protocols = args.records

    logger.train(f"Starting ID imputation for {len(protocols)} protocols")

    num_ids = 0
    ids = []
    all_seed_maps = {}

    with multiprocessing.Pool() as pool:
        for i, n, smap in tqdm(
            pool.imap(add_protocol_id, protocols),
            total=len(protocols),
            desc="Processing protocols",
        ):
            ids.extend(i)
            num_ids += n
            all_seed_maps.update(smap)

    c = Counter(ids)
    duplicates = {id_val: count for id_val, count in c.items() if count > 1}

    if duplicates:
        logger.warning("Duplicate xml:id values detected")
        for id_val, count in list(duplicates.items())[:10]:
            logger.warning(
                f"{id_val} (used {count} times), seed: "
                f"{all_seed_maps.get(id_val, 'N/A')}"
            )
    else:
        logger.info("No duplicate xml:id values found")

    logger.info(f"Total IDs generated: {num_ids}")
    logger.info(f"Unique IDs: {len(set(ids))}")
    logger.info(f"Duplicate IDs: {len(duplicates)}")

    if len(set(ids)) != num_ids:
        logger.critical("Duplicate IDs detected — aborting")
        raise AssertionError("There are duplicate IDs!")

    logger.train("ID imputation completed successfully")


if __name__ == "__main__":
    parser = fetch_parser("records")
    args = impute_args(parser.parse_args())
    main(args)
