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
)
from pyriksdagen.utils import (
    get_data_location,
)
from tqdm import tqdm
import pandas as pd
import re




i_ort = re.compile(r'(i/från)\s(\S)+')
stray_i_ort = re.compile(r'^(från|i)\s\S+\s')
party_abbrev = re.compile(r'\((\S{1,4})\)')
end_initial = re.compile(r'.*\s[A-ZÀ-ÖØ-Þ]$')
start_initial = re.compile(r'^[A-ZÀ-ÖØ-Þ]\.')





def match_author(name, db, party_mapping):
    pa = None
    specifier = None
    m = party_abbrev.search(name)
    if m:
        name = name[:m.start()]
        pa = m.group(1).lower()
    m = None
    m = i_ort.search(name)
    if m:
        name = name[:m.start()]
        specifier = m.group(2).lower()
    d = {
        "name": name.lower(),
        "party_abbrev": pa,
        "specifier": specifier,
    }
    id = detect_mp(d, db, party_map=party_mapping)
    if id is None:
        id = detect_mp(d, db, match_fuzzily=True, party_map=party_mapping)
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

    for i, motion in enumerate(tqdm(args.motions)):
        py = motion.split("/")[2]
        if py in ["fort", "reg"]:
            continue
        root, ns = parse_tei(motion)
        sb = root.findall(f".//{ns['tei_ns']}p[@type=\"signatureBlock\"]")
        sb.extend(root.findall(f".//{ns['tei_ns']}div[@type=\"signatureBlock\"]"))
        for s in sb:
            t = ' '.join(_.strip() for _ in s.text.splitlines() if _.strip() != '')
            if len(t) > 0:
                l = len(t.split(' '))

                if l > 267:
                    del s.attrib["type"]
                    s.tag = "p"
                    continue
                else:
                    if l not in lens_counts:
                        lens_counts[l] = 0
                    lens_counts[l] += 1
                    lens.append([motion, l, t])
                #print('\n', t)
                names = handle_block_text(t)
                #print("4", names)
                #[print("  ", _) for _ in names]
                #print(motion, s.text.strip())
                s.tag = "div"
                s.text = None
                for name in names:
                    ne = etree.SubElement(s, "p")
                    ne.text=name
                    m = stray_i_ort.match(name)
                    if m and m.group(0) == name:
                        ne.attrib["type"] = "strayIOrt"
                    else:
                        ne.attrib["type"] = "signature"
                        ne.attrib["who"] = match_author(name, db, party_mapping)
            if len(s) > 0:
                for sig in s:
                    if "type" in sig.attrib:
                        if sig.attrib["type"] == "signature":
                            if args.redetect_knowns:
                                sig.attrib["who"] = match_author(sig.text, db, party_mapping)
                            else:
                                if "who" in sig.attrib and sig.attrib["who"] == "unknown":
                                    sig.attrib["who"] = match_author(sig.text, db, party_mapping)
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
    main(impute_args(parser.parse_args()))
