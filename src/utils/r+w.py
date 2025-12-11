#!/usr/bin/env python3
"""
Parse a corpus and write it back to disk.
"""
from pyriksdagen.args import (
    fetch_doctype_parser,
    impute_args,
)
from pyriksdagen.io import (
    parse_tei,
    write_tei,
)
from tqdm import tqdm
import sys




def main(args):
    argd = vars(args)
    for doc in tqdm(argd[argd["doctype"]]):
        write_tei(parse_tei(doc, get_ns=False), doc)




if __name__ == '__main__':
    parser, argstring = fetch_doctype_parser(sys.argv, docstring=__doc__)
    main(impute_args(parser.parse_args(argstring)))


