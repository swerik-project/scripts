#!/usr/bin/env python3
"""
List absurdly long ranges of multimotion pages.
"""
from common.xml_utils import (
    write_xml,
)
from datetime import datetime
from glob import glob
from lxml import etree
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from pyriksdagen.utils import (
    get_formatted_uuid,
    parse_tei,
)
from tqdm import tqdm
import argparse, os
import pandas as pd
import regex as re



def list_years(args):
    args = vars(args)
    print(args)
    if args["parliament_year"] is not None:
        years = args["parliament_year"]
    elif args["start"] is not None:
        _range = [_ for _ in os.listdir(args["data_folder"]) if os.path.isdir(f"{args['data_folder']}/{_}") and _ not in ["fort", "reg"]]
        years = sorted([_ for _ in _range if args['start'] <= int(_[:4]) <= args['end']])
    else:
        raise Error("Gah... I don't know what to do! Did you set start/end, year or pass alto packages?")
    return years


def main(args):
    orcid = None

    pat = re.compile(r'(\S{0,3}\s?(Nr\s[0-9-–—\.(B\s?)]+\s)?([-–—=]\s)?Motion(er)?\s(i|till)\s(Andra|Första)\skammaren,\snr\s((B\s?)?[0-9\.]+([-–—]{1,2}(B\s?)?[0-9\.]+))(\s(å)r\s[0-9-\/\.]{4,9}\s?\S{0,3})?){i<=2,d<=2,s<=2,e<=3}')

    years = list_years(args)
    for year in years:
        print(year)
        if args.list:
            rows = []
            cols = ["mot", "fw_text", "range", "fw_id"]
            print(year)
            c = 0
            no_match = 0
            year_D = {}
            motions = sorted(glob(f"{args.data_folder}/{year}/*.xml"))
            for mot in tqdm(motions):
                print(mot)
                mot_refs = []
                root, ns = parse_tei(mot, get_ns=True)
                fws = root.findall(f".//{ns['tei_ns']}fw")
                for fw in fws:
                    m = pat.match(' '.join([_.strip() for _ in fw.text.splitlines() if _.strip() != '']))
                    if m is not None:
                        _nr = m.group(7)
                        nr = _nr.strip()
                        is_range = False
                        is_B = False
                        for _ in "-–—":
                            if _ in nr:
                                is_range = True
                        if "B" in nr:
                            is_B = True
                            nr = nr.replace("B", "")
                            nr = nr.replace(" ", "")
                        if is_range:
                            print([_.strip() for _ in re.split('\D', nr) if _.strip() != ''])
                            try:
                                start, end = [_.strip() for _ in re.split('\D', nr) if _.strip() != '']
                            except:
                                rows.append([mot,  " ".join([_.strip() for _ in fw.text.splitlines() if _.strip() != '']), "~~", fw.attrib[f"{ns['xml_ns']}id"]])
                            else:
                                if len(list(range(int(start), int(end)+1))) > 5:
                                    print(mot, len(list(range(int(start), int(end)+1))))
                                    rows.append([mot, " ".join([_.strip() for _ in fw.text.splitlines() if _.strip() != '']), f"{start}-{int(end)+1}", fw.attrib[f"{ns['xml_ns']}id"]])

            df = pd.DataFrame(rows, columns=cols)
            df.to_csv(f"{args.io_path}/_{year}-absurd-ranges.tsv", sep='\t', index=False)

        if args.fix_listed:
            if orcid is None:
                orcid = os.environ.get("ORCID")
                if orcid is None:
                    orcid = input("Enter your ORC ID: ")
            date = datetime.now().strftime('%Y-%m-%d')
            try:
                df = pd.read_csv(f"{args.io_path}/_{year}-absurd-ranges_corrected.tsv", sep='\t')
            except:
                print("no corrections file for", year)
            else:
                for i, r in tqdm(df.iterrows(), total=len(df)):
                    root, ns = parse_tei(r['mot'], get_ns=True)
                    try:
                        revisionDesc = root.find(f".//{ns['tei_ns']}revisionDesc")
                        assert revisionDesc is not None
                    except:
                        revisionDesc = etree.SubElement(root.find(f"{ns['tei_ns']}teiHeader"), "revisionDesc")
                    _text = r["fw_text"].split('ə')
                    fw_elem = root.find(f".//{ns['tei_ns']}fw[@{ns['xml_ns']}id=\"{r['fw_id']}\"]")
                    fw_elem.text = _text[0]
                    correction = etree.SubElement(revisionDesc,
                                                "correction",
                                                attrib={"who": f"orcid_{orcid}",
                                                        "when": date,
                                                        "corresp": r['fw_id']})
                    correction.text = "OCR correction"
                    if len(_text) > 1:
                        seedtext = r['mot']
                        parent = fw_elem.getparent()
                        if parent.tag == f"{ns['tei_ns']}body":
                            parent = root.find(f".//{ns['tei_ns']}div[@type=\"motBody\"]/{ns['tei_ns']}div")
                            fwi = 0
                        else:
                            fwi = parent.index(fw_elem) + 1
                        for ix, _text_elem in enumerate(_text[1:]):
                            p = etree.Element("p")
                            p.text = _text_elem
                            seedtext += _text_elem
                            p_id = get_formatted_uuid(seed=seedtext)
                            p.attrib[f"{ns['xml_ns']}id"] = p_id
                            parent.insert(fwi+ix, p)
                            correction = etree.SubElement(revisionDesc,
                                                        "correction",
                                                        attrib={"who": f"orcid_{orcid}",
                                                                "when": date,
                                                                "corresp": p_id})
                            correction.text = "OCR correction"
                    write_xml(root, r['mot'])



if __name__ == '__main__':
    parser = fetch_parser("motions", docstring=__doc__)
    parser.add_argument("-o", "--io-path", default="input/mot-ranges")
    parser.add_argument("--list", action='store_true')
    parser.add_argument("--fix-listed", action='store_true')
    main(impute_args(parser.parse_args()))
