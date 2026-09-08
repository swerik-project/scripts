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
from pyriksdagen.db import filter_db

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



i_ort = re.compile(r'(i/från)\s(\S)+')
stray_i_ort = re.compile(r'^(från|i)\s\S+\s')
party_abbrev = re.compile(r'\((\S{1,4})\)')
end_initial = re.compile(r'.*\s[A-ZÀ-ÖØ-Þ]$')
start_initial = re.compile(r'^[A-ZÀ-ÖØ-Þ]\.')





def match_author(name, db, party_mapping):
    d = intro_to_dict(name)
    id = detect_mp(d, db, party_map=party_mapping)
    #if id is None:
    #    id = detect_mp(d, db, match_fuzzily=True, party_map=party_mapping)
    #    print("step3")
    if id is None:
        return "unknown"
    return id


def flatten_list(l):
    return list(chain.from_iterable([[_] if type(_) is not list else _ for _ in l]))


def handle_block_text(t):
    names = [_.strip() for _ in t.split(")")]
    if len(names) > 1:
        names = [f"{_})" for _ in names]
    #print("1", names)
    for i, name in enumerate(names):
        m = stray_i_ort.match(name)
        if m is not None:
            #print(m, m.start(), m.end())
            names[i] = [name[:m.end()], name[m.end():]]
    names = flatten_list(names)
    for i, name in enumerate(names):
        split_names = []
        name_s = [_.strip() for _ in name.split(",")]
        names[i] = name_s
    names = flatten_list(names)
    #print("3", names)
    for i, name in enumerate(names):
        #print("n", name)
        initials = None
        split_names = []
        name_s = [_.strip() for _ in name.split(".") if _.strip() != ""]
        for _ in name_s:
            _ = _.strip()
            if 0 < len(_) < 3:
                if not initials:
                    initials = f"{_}."
                else:
                    initials = initials + ' ' + f"{_}."
            else:
                if initials and len(initials) > 0:
                    split_names.append(f"{initials} {_}")
                    initials = None
                else:
                    split_names.append(_.strip())
        names[i] = split_names
    names = flatten_list(names)

    for i, name in enumerate(names):
        m = end_initial.match(name)
        if m and i+1 < len(names):
            names[i] = name + ' ' + names[i+1]
            names[i+1] = ""
    names = [_.strip() for _ in names if _ != ""]
    for i, name in enumerate(names):
        s = name.split(' ')
        if len(s) == 1 and i+1 < len(names):
            m = start_initial.match(names[i+1])
            if m:
                names[i] = name + ' ' + names[i+1]
                names[i+1] = ""
            else:
                ss = names[i+1].split(' ')
                if len(ss) == 1:
                    names[i] = name + ' ' + names[i+1]
                    names[i+1] = ""
        elif len(s) == 1 and i+1 == len(names):
            names[i-1] = names[i-1] + ' ' + name
            names[i] = ""
    names = [_.strip() for _ in names if _ != ""]
    names = flatten_list(names)
    return names




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
                            item.attrib["who"] = match_author(t, db_year, party_mapping)
                        elif item.attrib.get("who") == "unknown":
                            item.attrib["who"] = match_author(t, db_year, party_mapping)
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
