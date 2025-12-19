#!/usr/bin/env python3
"""
Curation of motions from different sources led to inconsistencies in the signature block structure and attribute
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
        write = False
        root, ns = parse_tei(motion)
        blocks = root.findall(f".//{ns['tei_ns']}div[@type=\"signatureBlock\"]")
        for block in blocks:
            if any(_.tag.endswith("list") for _ in block):
                raise ValueError("This shouldn't happen")
            list_ = etree.Element("list")
            for p in block:
                list_.append(p)
            for p in list(block):
                block.remove(p)
            for item in list_:
                item.tag = "item"
            block.append(list_)
            write = True
        signs = root.findall(f".//{ns['tei_ns']}div[@type=\"motSignatures\"]")
        for sign in signs:
            sign.attrib["type"] = "signatureBlock"
            for list_ in sign:
                for item in list_:
                    item.attrib["type"] = "signature"
            write = True
        if write:
            write_tei(root, motion)




if __name__ == '__main__':
    parser = fetch_parser("motions", docstring=__doc__)
    main(impute_args(parser.parse_args()))
