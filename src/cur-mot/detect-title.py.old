#!/usr/bin/env python3
"""
Detect titles in motions.
"""
from lxml import etree
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from pyriksdagen.utils import (
    parse_tei,
)
from tqdm import tqdm

def main(args):
    T = 0
    t = 0
    f = 0
    for motion in tqdm(args.motions):
        root, ns = parse_tei(motion)
        if root is None:
            print("NO ROOT!")
        try:
            meta_title = root.find(f".//{ns['tei_ns']}titleStmt/{ns['tei_ns']}title")
            assert meta_title != None
        except:
            meta_title = root.find(f".//titleStmt/title")

        try:
            mot_title = root.find(f".//{ns['tei_ns']}div[@type=\"motTitle\"]")
            assert mot_title != None
        except:
            mot_title = root.find(f".//div[@type=\"motTitle\"]")
        if meta_title is not None:
            T += 1
            if mot_title is not None:
                #print(meta_title.text, "--", mot_title.text, "--", mot_title.text == meta_title.text)
                if mot_title.text == meta_title.text:
                    t += 1
            else:
                #print("XXX", motion, meta_title.text)
                f += 1
        else:
            #print(root)
            for elem in root.iter():
                if elem.tag.endswith("}p"):
                    #print(elem.text.strip())
                    #rint(elem.tag)
                    if elem.text.strip().startswith('Stockholm den '):
                        print(elem.text.strip())
            try:
                dates = root.xpath(f".//{ns['tei_ns']}p[starts-with(normalize-space(string()), 'Stockholm den')]")
                assert dates
            except:
                dates = root.xpath(".//p[starts-with(normalize-space(string()), 'Stockholm den')]")

            for date in dates:
                print(date.text.strip())







    print("Motions:", len(args.motions))
    print("Title in metadata:", T)
    print("Body title == Meta title (T, F):", t, f)






if __name__ == '__main__':
    parser = fetch_parser("motions", docstring=__doc__)
    main(impute_args(parser.parse_args()))
