#!/usr/bin/env python3
"""
Check that the number (filename) is actually in the document.
"""
from common.args import (
    alto_args,
    list_years,
    verify_alto_args,
)
from common.xml_utils import (
    parse_xml,
    write_xml,
)
from glob import glob
from lxml import etree
from tqdm import tqdm
import argparse, os
import pandas as pd
import regex as re


def main(args):
    years = list_years(args)
    for year in years:
        print(year)
        if args.list:
            debug = False
            p_count = 0
            matches = 0
            mismatch = 0
            mismatch_2 = 0
            no_match = 0
            motions = sorted(glob(f"{args.motionspath}/{year}/*.xml"))
            rows = []
            cols = ["mot", "elem_id", "nr", "text"]

            for mot in tqdm(motions):
                if mot == "riksdagen-motions/data/1962/mot-1962--ak--00262.xml":
                    debug = True
                else:
                    debug = False
                root, ns = parse_xml(mot, get_ns=True)
                body = root.find(f".//{ns['tei_ns']}body")

                nr = int(mot.split('-')[-1].replace(".xml", ""))
                pat = re.compile(fr'((N|n)r[\.\,]?\s{nr}[\.\,]?){{i<=1,d<=1,s<=1,e<=1}}')
                pat_2 = re.compile(r'((N|n)r[\.\,]?\s\S+){i<=1,d<=1,s<=1,e<=1}')
                pat_3 = re.compile(r'(N|n)r[\.\,]?\s\S+')
                ps = root.findall(f".//{ns['tei_ns']}p")
                M = False
                for p in ps:
                    p_count += 1
                    _text = " ".join([_.strip() for _ in p.text.splitlines() if _.strip() != ''])
                    if debug: print("0", _text)
                    m = None
                    m = pat.match(_text)
                    if m is not None:
                        matches += 1
                        M = True
                    else:
                        if debug: print("1", _text)
                if not M:
                    for p in ps:
                        _text = " ".join([_.strip() for _ in p.text.splitlines() if _.strip() != ''])
                        m = None
                        m = pat_2.match(_text)
                        if m is not None:
                            mismatch += 1
                            #M = True
                            rows.append([mot, p.attrib[f"{ns['xml_ns']}id"], nr, m.group(0)])
                        else:
                            if debug: print("2", _text)
                if not M:
                    for p in ps:
                        _text = " ".join([_.strip() for _ in p.text.splitlines() if _.strip() != ''])
                        m = None
                        m = pat_3.search(_text)
                        if mot == "riksdagen-motions/data/1962/mot-1962--ak--00262.xml":
                            print(m)
                        if m is not None:
                            mismatch_2 += 1
                            M = True
                            rows.append([mot, p.attrib[f"{ns['xml_ns']}id"], nr, _text])
                        else:
                            if debug: print("3", _text)
                    if not M:
                        no_match += 1
                        rows.append([mot, None, nr, None])
            df = pd.DataFrame(rows, columns=cols)
            df.drop_duplicates(inplace=True)
            df.to_csv(f"{args.io_path}/_{year}-nomatch-nr.tsv", sep="\t", index=False)


            print("  p", p_count)
            print("  ma", matches)
            print("  mi", mismatch)
            print("  mi2", mismatch_2)
            print("  0", no_match)

        if args.fix_listed:
            pass




if __name__ == '__main__':
    parser = alto_args(__file__)
    parser.add_argument("-o", "--io-path", default="input/mot-unmatched-nr")
    parser.add_argument("--list", action='store_true')
    parser.add_argument("--fix-listed", action='store_true')
    args = parser.parse_args()
    main(verify_alto_args(args))
