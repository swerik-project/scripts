#!/usr/bin/env python3
"""
Untag stray i-ort as type=signature in signature block
"""
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from pyriksdagen.io import (
    parse_tei,
    write_tei,
)
from tqdm import tqdm
import re




def main(args):
    stray_iort = re.compile(r'^(i|från)\s\S+(\s\S)?$')
    for motion in tqdm(args.motions):
        write = False
        root, ns = parse_tei(motion)
        signatures = root.findall(f".//{ns['tei_ns']}p[@type=\"signature\"]")
        signatures.extend(root.findall(f".//{ns['tei_ns']}item[@type=\"signature\"]"))
        for signature in signatures:
            txt = ' '.join([_.strip() for _ in signature.text.splitlines() if _.strip() != ""])
            m = stray_iort.match(txt)
            if m and "type" in signature.attrib and signature.attrib["type"] == "signature":
                del signature.attrib["type"]
                if "who" in signature.attrib and signature.attrib["who"] == "unknown":
                    del signature.attrib["who"]
                write = True
        if write:
            write_tei(root, motion)




if __name__ == '__main__':
    parser = fetch_parser("motions", docstring=__doc__)
    main(impute_args(parser.parse_args()))
