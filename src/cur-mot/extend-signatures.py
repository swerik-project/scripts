#!/usr/bin/env python3
"""
Extend signature block if it is preceded by multiple last names
"""
from itertools import chain
from lxml import etree
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from pyriksdagen.io import (
    parse_tei,
    write_tei,
)

from pyriksdagen.utils import (
    get_data_location,
    infer_metadata,
    first_and_last_names,
    TEI_NS,
    XML_NS
)
from tqdm import tqdm
import polars as pl
import re
from trainerlog import get_logger

LOGGER = get_logger("map-signatures")

def move_elem_to_sigblock(root, elem, sigblock):
    sigblock_list = sigblock.find(f"{TEI_NS}list")
    if sigblock_list is None:
        record_id = root.get(f"{XML_NS}id")
        LOGGER.error(f"Signature block {sigblock} (len {len(sigblock)}) in {record_id} does not contain a list")
        if len(sigblock) == 0:
            LOGGER.error(f"Signature block {sigblock} text? {sigblock.text}")
            sigblock_list = etree.SubElement(sigblock, f"{TEI_NS}list")
        
    elem.tag = "item"
    elem.attrib["who"] = "unknown"
    elem.attrib["type"] = "signature"
    sigblock_list.append(elem)
    return root

def main(args):
    metadata_location = get_data_location("metadata")

    df_names = pl.read_csv(f'{metadata_location}/name.csv')
    df_iort = pl.read_csv(f'{metadata_location}/location_specifier.csv')
    first_names, last_names, iort = first_and_last_names(df_names, df_iort)
    all_names_etc = first_names.union(last_names).union(iort) - {"Stockholm"}
    new_blocks = 0
    # Pre-load regex for intro_to_dict for performance reasons
    for i, motion in enumerate(tqdm(sorted(args.motions))):
        py = motion.split("/")[2]
        if py in ["fort", "reg"]:
            continue
        root, ns = parse_tei(motion)

        # Filter db to only include MPs that had a mandate that year
        metadata = infer_metadata(motion)

        blocks = root.findall(f".//{ns['tei_ns']}signatureBlock")
        for signatureBlock in blocks:
            item = signatureBlock.getnext()
            if item is not None and item.text is not None:
                t = ' '.join(item.text.split())
                if len(t) > 0:
                    no_wds = len(t.split())
                    wds_last_names = [wd.replace(".", "") in all_names_etc for wd in t.split()]
                    multiple_names = sum(wds_last_names) >= 2
                    long_para = no_wds >= 20 and sum(wds_last_names) / no_wds <= 0.2
                    ad_hoc = "tryckeri" not in t.lower() and "aktie" not in t.lower()
                    if multiple_names and not long_para and ad_hoc:
                        new_blocks += 1
                        LOGGER.info(f"Multiple names: {t} in {motion}")
                        move_elem_to_sigblock(root, item, signatureBlock)
                        #split_signature(item, first_names, last_names, iort)

                    elif multiple_names:
                        LOGGER.warning(f"Multiple names but long paragraph: {t} ({sum(wds_last_names)} / {no_wds})")

        write_tei(root, motion)

    LOGGER.train(f"New signature blocks: {new_blocks} in {len(args.motions)}, i.e. {(1000 * new_blocks // len(args.motions))/10}%")


if __name__ == '__main__':
    parser = fetch_parser("motions", docstring=__doc__)
    parser.add_argument("--redetect-knowns", action='store_true')
    parser.add_argument("--metadata-location",
                        type=str,
                        default="input/metadata/db.pkl",
                        help="path to compiled metadata db")
    parser.add_argument("--recompile-metadata", action='store_true')
    parser.add_argument("--write-compiled-db",
                        type=str,
                        default="True",
                        choices=["True", "False"],
                        help="write db after compile")

    args = parser.parse_args()
    LOGGER.info(f"args: {args}")
    main(impute_args(args))
