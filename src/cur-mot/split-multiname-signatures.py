#!/usr/bin/env python3
"""
Map names in signature blocks to metadata.
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
    first_and_last_names
)
from tqdm import tqdm
import polars as pl
import re
from trainerlog import get_logger

LOGGER = get_logger("map-signatures")

def split_signature(elem, first_names, last_names, iort):
    t = ' '.join(elem.text.split())
    t_before = t
    words = t.split()
    for w0, w1 in zip(words[:-1], words[1:]):
        if "i" not in [w0,w1]:
            cond1 = w0 in last_names.union(iort)
            cond1 = cond1 or (w0[0].isupper() and "son" == w0[-3:])
            cond2 = w1 in first_names or (w1 not in last_names and w1[0].isupper())

            if cond1 and cond2:
                LOGGER.debug(f"Split between : {w0} and {w1}")
                t = t.replace(f"{w0} {w1}", f"{w0}; {w1}")
    #exit()
    if t_before != t:
        LOGGER.info(f"{t_before} => {t}")

    return


def main(args):
    metadata_location = get_data_location("metadata")

    df_names = pl.read_csv(f'{metadata_location}/name.csv')
    df_iort = pl.read_csv(f'{metadata_location}/location_specifier.csv')
    first_names, last_names, iort = first_and_last_names(df_names, df_iort)

    with open("first_names.txt", "w") as f:
        f.write("\n".join(list(first_names)))
    with open("last_names.txt", "w") as f:
        f.write("\n".join(list(last_names)))
    with open("iort.txt", "w") as f:
        f.write("\n".join(list(iort)))

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
            for item in signatureBlock.findall(f".//{ns['tei_ns']}item"):
                t = ' '.join(item.text.split())
                if len(t) > 0:
                    if item.attrib.get("type") == "signature":
                        multiple_names = [wd in last_names for wd in t.split()]
                        multiple_names = sum(multiple_names) >= 2
                        if multiple_names:
                            #LOGGER.warning(f"Multiple names: {t}")
                            split_signature(item, first_names, last_names, iort)

        #exit()
        #write_tei(root, motion)
    df = pd.DataFrame(lens, columns = ["motion", "length_of_sig_block", "sig_block_text"])
    df.to_csv("input/motion_sig_block_len.csv", index=False)
    #{print(k, v) for k, v in dict(sorted(lens_counts.items(), key=lambda item: item[1])).items()}




if __name__ == '__main__':
    parser = fetch_parser("motions", docstring=__doc__)
    parser.add_argument("--redetect-knowns", action='store_true')
    parser.add_argument("--skip-multiple-names", action='store_true')
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
