#!/usr/bin/env python3
"""
take a sample of N documents per year.
"""
from datetime import datetime
from glob import glob
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from pyriksdagen.utils import get_data_location
import argparse
import hashlib
import numpy as np
import os
import pandas as pd
import random


def get_random_state(args, data_path):
    if args.seed:
        digest = hashlib.md5(args.seed.encode("utf-8")).digest()
        digest = int.from_bytes(digest, "big") % (2**32)
        random_state = np.random.RandomState( (int(digest)+int(args.seed)) % (2**32))
    else:
        digest = hashlib.md5(data_path.encode("utf-8")).digest()
        digest = int.from_bytes(digest, "big") % (2**32)
        random_state = np.random.RandomState( (int(digest)+int(year)) % (2**32))
    return random_state


def main(args):
    sampled_documents = []
    data = get_data_location(args.doctype)
    print(data)
    years = [_ for _ in os.listdir(data) if os.path.isdir(f"{data}/{_}") and _ not in ["fort", "reg"]]
    for year in years:
        print(year)
        populations = []
        random_state = get_random_state(args, f"{data}/{year}")
        if int(year) <= 1970 and args.doctype == "records":
            populations.append(glob(f"{data}/{year}/*-ak-*.xml"))
            populations.append(glob(f"{data}/{year}/*-fk-*.xml"))
        else:
            populations.append(glob(f"{data}/{year}/*.xml"))
        for population in populations:
            sample = random_state.choice(np.array(population), size=args.number_to_sample_per_year)
            sampled_documents.extend(list(sample))
    print(sorted(sampled_documents), len(sampled_documents))
    outfilename = "docdate"
    if args.doctype =="motions":
        outfilename = outfilename + "-titles"
        df = pd.DataFrame([[_,None,None] for _ in sampled_documents], columns=["motion", "docdate", "title"])
        sortcol = "motion"
    else:
        df = pd.DataFrame([[_,None] for _ in sampled_documents], columns=["record", "docdate"])
        sortcol = "record"
    if args.seed is not None:
        outfilename = outfilename + f"_seed+{args.seed}_qeAnnotations.csv"
    else:
        outfilename = outfilename + f"__qeAnnotations.csv"
    outpath = f"riksdagen-{args.doctype}/quality/data/{outfilename}"
    df.sort_values(by=[sortcol], inplace=True)
    df.to_csv(outpath, index=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--doctype", required=True, choices=["records", "motions"])
    parser.add_argument("-n", "--number-to-sample-per-year", type=int)
    parser.add_argument("--generate-seed", action="store_true")
    parser.add_argument("--seed", default=None)
    args=parser.parse_args()
    if args.generate_seed:
        args.seed = datetime.now().strftime("%Y%m%d%H%M%S")
    main(args)
