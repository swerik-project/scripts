#!/usr/bin/env python3
"""
Assume all blocks after a date element is a signature
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
from tqdm import tqdm





def main(args):
    for motion in tqdm(args.motions):
        root, ns = parse_tei(motion, get_ns=True)
        date_elems = root.findall(f".//{ns['tei_ns']}p[@type=\"date\"]")
        for de in date_elems:
            n = de.getnext()
            if n is not None and ("type" not in n.attrib or n.attrib["type"] != "signatureBlock"):
                n.attrib["type"] = "signatureBlock"
        write_tei(root, motion)




if __name__ == '__main__':
    parser = fetch_parser("motions", docstring=__doc__)
    main(impute_args(parser.parse_args()))
