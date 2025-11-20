#!/usr/bin/env python3
"""
Read and write the records corpus.
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




def main(args):
    for record in tqdm(args.records):
        root, ns = parse_tei(record)
        write_tei(root, record)




if __name__ == '__main__':
    parser = fetch_parser("records", docstring=__doc__)
    main(impute_args(parser.parse_args()))
