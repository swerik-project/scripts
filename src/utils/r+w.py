#!/usr/bin/env python3
"""
Parse a corpus and write it back to disk.
"""
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
    interpellation_parser,
    motion_parser,
    record_parser,
    volg_parser,
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

    def _errmahgerd(err, corpora):
        errs = {
                1: "You didn't set any first argument.",
                2: "You passed an invalid doctype."
            }
        raise ValueError(f"You have to pass the document type as the first argument of this script ({' '.join(['--'+k+',' for k in corpora.keys()])})\n{errs[err]}")

    corpora = {
            "records": record_parser,
            "motions": motion_parser,
            "interpellations": interpellation_parser,
            "volg": volg_parser,
        }
    if not sys.argv[1]:
        _errmahgerd(1, corpora)
    if sys.argv[1][2:] in corpora:
        parser = fetch_parser(sys.argv[1][2:], docstring=__doc__)
        main(impute_args(parser.parse_args(sys.argv[2:])))
    else:
        _errmahgerd(2, corpora)
