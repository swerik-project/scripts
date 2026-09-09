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
from pyriksdagen.metadata import (
    load_Corpus_metadata,
)
from pyriksdagen.segmentation import (
    detect_mp,
    intro_to_dict
)
from pyriksdagen.db import filter_db, load_expressions

from pyriksdagen.utils import (
    get_data_location,
    infer_metadata
)
from tqdm import tqdm
import pandas as pd
import re
from trainerlog import get_logger
import datetime

LOGGER = get_logger("map-signatures")

def match_author(name, db, party_mapping, match_fuzzily=False, expressions=None):
    d = intro_to_dict(name, expressions=expressions)
    if len(d) == 0:
        return "unknown"

    id = detect_mp(d, db, party_map=party_mapping)
    if id is None and match_fuzzily:
        id = detect_mp(d, db, match_fuzzily=True, party_map=party_mapping)
    if id is None:
        return "unknown"
    return id


def main(args):
    lens_counts = {}
    lens = []
    metadata_location = get_data_location("metadata")
    party_mapping = pd.read_csv(f'{metadata_location}/party_abbreviation.csv')
    if args.recompile_metadata:
        db = load_Corpus_metadata()
        db.rename(columns={"person_id":"id", "location":"specifier"}, inplace=True)
        db['name'] = db['name'].apply(lambda x: x.lower().strip())
        db['start'] = db['start'].apply(lambda x: x if type(x)==str else x.strftime('%Y-%m-%d'))
        db['end'] = db['end'].apply(lambda x: x if type(x)==str else x.strftime('%Y-%m-%d'))
        if args.write_compiled_db == "True":
            db.to_pickle(args.metadata_location)
    else:
        db = pd.read_pickle(args.metadata_location)

    db['start'] = pd.to_datetime(db['start'])
    db['end'] = pd.to_datetime(db['end'])
    year = None
    db_year = None

    # Pre-load regex for intro_to_dict for performance reasons
    intro_expressions = load_expressions(phase="mp")
    for i, motion in enumerate(tqdm(sorted(args.motions))):
        py = motion.split("/")[2]
        if py in ["fort", "reg"]:
            continue
        root, ns = parse_tei(motion)

        # Filter db to only include MPs that had a mandate that year
        metadata = infer_metadata(motion)
        if year != metadata["year"]:
            start_date = datetime.datetime(metadata["year"], 1, 1)
            end_date = datetime.datetime(metadata["year"]+1, 12, 31)
            db_year = filter_db(db, start_date=start_date, end_date=end_date)
            year = metadata["year"]

        blocks = root.findall(f".//{ns['tei_ns']}signatureBlock")
        for signatureBlock in blocks:
            for item in signatureBlock.findall(f".//{ns['tei_ns']}item"):
                t = ' '.join(item.text.split())
                if len(t) > 0:
                    if item.attrib.get("type") == "signature":
                        if args.redetect_knowns:
                            item.attrib["who"] = match_author(t, db_year, party_mapping, expressions=intro_expressions)
                        elif item.attrib.get("who") == "unknown":
                            item.attrib["who"] = match_author(t, db_year, party_mapping, expressions=intro_expressions)
        write_tei(root, motion)
    df = pd.DataFrame(lens, columns = ["motion", "length_of_sig_block", "sig_block_text"])
    df.to_csv("input/motion_sig_block_len.csv", index=False)
    #{print(k, v) for k, v in dict(sorted(lens_counts.items(), key=lambda item: item[1])).items()}




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
