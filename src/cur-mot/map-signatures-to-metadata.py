#!/usr/bin/env python3
"""
Map names in signature blocks to metadata.
"""
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
    infer_metadata,
    first_and_last_names
)
from tqdm import tqdm
import polars as pl
import re
from trainerlog import get_logger
import datetime

LOGGER = get_logger("map-signatures")
NAME_REGEX = re.compile('^[A-ZÀ-Þ\\-][a-zß-ÿ\\-]{0,15} ([A-ZÀ-Þ\\-][a-zß-ÿ\\-]{0,15} )?[A-ZÀ-Þ\\-][a-zß-ÿ\\-]{1,15}')
LAST_NAME_REGEX = re.compile('^[A-ZÀ-Þ\\-][a-zß-ÿ\\-]{0,15}$')

def match_author(name, db, party_mapping, match_fuzzily=False, expressions=None):
    name = name.replace(".", "")
    d = intro_to_dict(name, expressions=expressions)
    LOGGER.debug(f"Name: {name}, dict: {d}")
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
    party_mapping_df = pl.read_csv(f'{metadata_location}/party_abbreviation.csv')
    party_mapping = {}
    for row in party_mapping_df.select("party", "abbreviation").to_dicts():
        # For some reason, the column names are weird
        # 'party' contains abbreviations, party names and 
        party_mapping[row["party"]] = row["abbreviation"]

    df_names = pl.read_csv(f'{metadata_location}/name.csv')
    df_iort = pl.read_csv(f'{metadata_location}/location_specifier.csv')
    first_names, last_names, iort = first_and_last_names(df_names, df_iort)
    print(len(first_names), len(last_names), len(iort))
    if args.recompile_metadata:
        db = load_Corpus_metadata()
        db = db.rename({"person_id":"id", "location":"specifier"})
        if args.write_compiled_db == "True":
            db.write_parquet(args.metadata_location)
    else:
        db = pl.read_parquet(args.metadata_location)

    LOGGER.info(f"Got db:\n{db}")

    year = None
    db_year = None

    # Pre-load regex for intro_to_dict for performance reasons
    intro_expressions = load_expressions(phase="mp")

    # Add custom expression for plain names
    name_exp_index = [t for _, t in intro_expressions].index("name")
    intro_expressions.insert(name_exp_index, (NAME_REGEX, "name"))
    intro_expressions.append((LAST_NAME_REGEX, "name"))
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
                        multiple_names = [wd in last_names for wd in t.split()]
                        multiple_names = sum(multiple_names) >= 2
                        if multiple_names:
                            LOGGER.warning(f"Multiple names: {t}")

                        skip_multiple_names = args.skip_multiple_names and multiple_names
                        if args.redetect_knowns and not skip_multiple_names:
                            item.attrib["who"] = match_author(t, db_year, party_mapping, match_fuzzily=args.fuzzy_matching, expressions=intro_expressions)
                        elif item.get("who") == "unknown" and not skip_multiple_names:
                            item.attrib["who"] = match_author(t, db_year, party_mapping, match_fuzzily=args.fuzzy_matching, expressions=intro_expressions)

                            if item.attrib["who"] != "unknown":
                                mp_id = item.attrib["who"]
                                LOGGER.info(f"MP matched: {mp_id} ({t})")
        write_tei(root, motion)




if __name__ == '__main__':
    parser = fetch_parser("motions", docstring=__doc__)
    parser.add_argument("--redetect-knowns", action='store_true')
    parser.add_argument("--skip-multiple-names", action='store_true')
    parser.add_argument("--fuzzy-matching", action='store_true')
    parser.add_argument("--metadata-location",
                        type=str,
                        default="input/metadata/db.parquet",
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
