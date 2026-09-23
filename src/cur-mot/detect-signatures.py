#!/usr/bin/env python3
"""
Aggressively detect signature blocks in motions where none are present
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
    get_formatted_uuid,
    TEI_NS,
    XML_NS
)
from tqdm import tqdm
import polars as pl
import re
from trainerlog import get_logger

LOGGER = get_logger("detect-signatures")

EXP = re.compile(
                 r"((S[A-ZÀ-Þa-zß-ÿ]{4,10}) (den [0-9]{1,2}|i) [A-ZÀ-Þ]?[a-zß-ÿ]{3,10} [0-9]{4}(.)? )"
                 r"([A-ZÀ-Þ][a-zß-ÿ]{0,15}. "
                 r"([A-ZÀ-Þ][a-zß-ÿ]{0,15}. )?"
                 r"[A-ZÀ-Þ][a-zß-ÿ]{1,15}(-)?[A-ZÀ-Þ]{0,1}[a-zß-ÿ]{0,15})"
                 )

EXP2 = re.compile(
                 r"([A-ZÀ-Þ][A-ZÀ-Þa-zß-ÿ]{3,10}) (den [0-9]{1,2}|i) [A-ZÀ-Þ]?[a-zß-ÿ]{3,10} [0-9]{4}(.)?$")

def convert_to_sigblock(elem, timestamp=None, name=None):
    t = ' '.join(elem.text.split())

    if name is not None:
        new_elem = etree.Element("p")
        parent = elem.getparent()
        ix = parent.index(elem)
        parent.insert(ix+1, new_elem)
        split_t = t.split(timestamp)
        assert len(split_t) == 2, f"Splitting by the 'Stockholm den ...' string should result in 2 parts, got: {split_t}"
        elem.text = split_t[0]
        new_elem.text = timestamp

        new_elem = etree.Element("p")
        parent = elem.getparent()
        ix = parent.index(elem)
        parent.insert(ix+2, new_elem)
        elem = new_elem

        split_t = t.split(name)
        assert len(split_t) <= 2
        elem.text = split_t[0]

        t = name
        if len(split_t) >= 2:
            t += split_t[1]



        

    original_id = elem.get(f"{XML_NS}id", get_formatted_uuid())
    elem.attrib[f"{XML_NS}id"] = get_formatted_uuid()
    LOGGER.info(f"Text: '{t}'")
    elem.text = None
    elem.tag = "signatureBlock"
    list_elem = etree.SubElement(elem, "list")
    list_elem.attrib[f"{XML_NS}id"] = get_formatted_uuid()
    item = etree.SubElement(list_elem, "item")
    item.text = t
    item.attrib[f"{XML_NS}id"] = original_id
    item.attrib["who"] = "unknown"
    item.attrib["type"] = "signature"


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
        pre_signature = False
        if root.find(f".//{ns['tei_ns']}signatureBlock") is None:
            body = root.find(f".//{ns['tei_ns']}body")
            for item in list(body.iter())[-args.tail_length:]:
                if item is not None and item.text is not None:
                    t = ' '.join(item.text.split())
                    if len(t) > 0:
                        no_wds = len(t.split())
                        wds_last_names = [wd.replace(".", "") in all_names_etc for wd in t.split()]
                        multiple_names = sum(wds_last_names) >= 2
                        long_para = sum(wds_last_names) / no_wds <= 0.3
                        ad_hoc = "tryckeri" not in t.lower() and "aktie" not in t.lower()

                        m = EXP.search(t)
                        m2 = EXP2.search(t)
                        if m is not None:
                            LOGGER.info(f"Found something {m.group(0)}")
                            LOGGER.info(f"Found something {m.groups()}")
                            timestamp = m.group(1)
                            name = m.group(5)
                            LOGGER.info(f"Name {name}")
                            
                            convert_to_sigblock(item, timestamp=timestamp, name=name)
                            write_tei(root, motion)

                            new_blocks += 1
                        elif m2 is not None:
                            LOGGER.info(f"Found something {m2.group(0)}")
                            pre_signature = True
                            new_blocks += 1

                        elif pre_signature:
                            convert_to_sigblock(item)
                            pre_signature = False

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
    parser.add_argument("--tail-length", type=int, default=15)
    parser.add_argument("--write-compiled-db",
                        type=str,
                        default="True",
                        choices=["True", "False"],
                        help="write db after compile")

    args = parser.parse_args()
    LOGGER.info(f"args: {args}")
    main(impute_args(args))
