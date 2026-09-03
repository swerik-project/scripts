#!/usr/bin/env python3

from pyriksdagen.args import (
    fetch_parser,
    impute_args
)

def abb(filename, list_):
    print(filename)
    abbr = filename.split('/')[-1].split('-')[2]
    if abbr is not None and abbr != '':
        if abbr not in list_:
            list_.append(abbr)
    return list_

def main(args):
    abb_a = []
    abb_b = []
    for m in args.motions:
        abbr_a = abb(m, abb_a)
    with open(args.tmp, 'r') as inf:
        files = inf.readlines()
        files = [_.strip() for _ in files if _.strip() != '']
        for f in files:
            abb_b = abb(f, abb_b)

    print(sorted(abb_a))
    print(sorted(abb_b))



if __name__ == '__main__':
    parser = fetch_parser("motions")
    parser.add_argument("--tmp", default="riksdagen-motions/_tmp.txt")
    args = parser.parse_args()
    main(impute_args(args))
