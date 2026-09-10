#!/usr/bin/env python3
"""
generate a lits of OCR corrections for test suite
"""
from common.args import (
    alto_args,
    list_years,
    verify_alto_args,
)
from common.xml_utils import (
    parse_xml,
)
from glob import glob
from lxml import etree
from tqdm import tqdm
import argparse, json
import pandas as pd




def main(args):
    years = list_years(args)
    D = {}
    cols = ["mot", "elem_id", "who", "when", "elem_tag", "elem_text"]
    for year in years:
        print(year)
        motions = sorted(glob(f"{args.motionspath}/{year}/*.xml"))
        #print(len(motions))
        for mot in tqdm(motions):
            #print(mot)
            root, ns = parse_xml(mot, get_ns=True)
            corrections = root.findall(f".//{ns['tei_ns']}correction")
            #if len(corrections) > 0:
            #    print("  ", len(corrections))
            for c in corrections:
                if c.text == "segment classification":
                    elem_id = c.attrib['corresp']
                    who = c.attrib['who']
                    when = c.attrib['when']
                    corresp = root.findall(f".//*[@{ns['xml_ns']}id=\"{elem_id}\"]")
                    if len(corresp) > 0:
            #            print(corresp)
                        elem_tag = corresp[0].tag
                        elem_attrib = dict(corresp[0].attrib)
                    else:
                        print("Elem not found!")
                    if mot not in D:
                        D[mot] = {}
                    if elem_id not in D[mot]:
                        D[mot][elem_id] = {}
                    D[mot][elem_id][when] = {"by":who, "elem_tag": elem_tag, "elem_attrib": elem_attrib}



    with open(f"{args.test_path}/segment_classification-corrections.json", 'w+') as out:
        json.dump(D, out, ensure_ascii=False, indent=4)




if __name__ == '__main__':
    parser = alto_args(__doc__)
    parser.add_argument("-o", "--test-path", default="riksdagen-motions/test/data")
    args = parser.parse_args()
    main(verify_alto_args(args))
