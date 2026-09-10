#!/usr/bin/env python3
"""
Classify headers and footers
"""
from common.xml_utils import (
    write_xml,
)
from glob import glob
from lxml import etree
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from pyriksdagen.utils import parse_tei
from tqdm import tqdm
import argparse, os
import regex as re




def main(args):
    pat = re.compile(r'((\S{0,3}\s?((N(r|:o)\s[0-9-–—\.(B\s?)]+\s)?([-–—=]\s)?Motion(e(r|n))?\s(i|till)\s(Andra|Första)\skammaren,|(Andra|Första)\skammarens\s(M|m)otioner)\sn(r|:o)\s[0-9-–—(B\s?)\.]+(\s(å)r\s[0-9-\/\.]{4,9}\s?\S{0,3})?){i<=2,d<=2,s<=2,e<=3}|(\S{1,3}\s)?(M|m)ot(\.|ion)\s\d{4}(\/(\d{2}|\d{4}))?:\s?\d{1,4}(\s\d{1,3})?)')

    for mot in tqdm(args.motions):
        fws_found = 0
        root, ns = parse_tei(mot)
        body = root.find(f".//{ns['tei_ns']}body")
        for elem in body.iter():
            if elem.text is not None:
                elem_text = " ".join([_.strip() for _ in elem.text.splitlines() if _.strip() != ''])
                if elem.text.strip() != '':
                    m = pat.match(elem_text)
                    if m is not None:
                        #print(m[0])
                        elem.tag = "fw"
                        elem.text = elem_text
                        if fws_found == 0:
                            body.insert(1, elem)
                        fws_found += 1
                    #else:
                    #    print(elem_text)
        if fws_found > 0:
            write_xml(root, mot)




if __name__ == '__main__':
    parser = fetch_parser("motions", docstring=__doc__)
    main(impute_args(parser.parse_args()))


