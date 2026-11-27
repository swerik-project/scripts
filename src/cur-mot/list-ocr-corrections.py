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
import argparse
import pandas as pd




def main(args):
    years = list_years(args)
    rows = []
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
                if c.text == "OCR correction":
                    elem_id = c.attrib['corresp']
                    who = c.attrib['who']
                    when = c.attrib['when']
                    corresp = root.findall(f".//*[@{ns['xml_ns']}id=\"{elem_id}\"]")
                    if len(corresp) > 0:
            #            print(corresp)
                        elem_tag = corresp[0].tag
                        elem_text = ' '.join([_.strip() for _ in corresp[0].text.splitlines() if _.strip != ''])
                    else:
                        print("Elem not found!")

                    rows.append([mot, elem_id, who, when, elem_tag, elem_text])

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(f"{args.test_path}/ocr-corrections.tsv", sep='\t', index=False)




if __name__ == '__main__':
    parser = alto_args(__doc__)
    parser.add_argument("-o", "--test-path", default="riksdagen-motions/test/data")
    args = parser.parse_args()
    main(verify_alto_args(args))
