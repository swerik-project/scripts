#!/usr/bin/env python3
"""
Add precision columns to csv files that have date start/end columns
"""
from pyriksdagen.utils import get_data_location
from tqdm import tqdm
import argparse
import os
import pandas as pd


def main(args):

    if args.data_location is None:
        args.data_location = get_data_location("metadata")

    if args.meta_file.endswith(".csv"):
        target_file = f"{args.data_location}/{args.meta_file}"
    else:
        target_file = f"{args.data_location}/{args.meta_file}.csv"

    if not os.path.exists(target_file):
        raise FileNotFoundError(target_file)

    for k, v in vars(args).items():
        print(k, v)

    df = pd.read_csv(target_file, sep=args.sep)
    df[f"{args.start_column}_precision"] = None
    df[f"{args.end_column}_precision"] = None
    for i,r in tqdm(df.iterrows(), total=len(df)):
        if pd.notnull(r[args.start_column]):# is not None:
            if len(r[args.start_column]) < 10 or r[args.start_column].endswith("-01-01"):
                df.at[i, f"{args.start_column}_precision"] = "year"
            else:
                df.at[i, f"{args.start_column}_precision"] = "day"
        if pd.notnull(r[args.end_column]):# is not None:
            if len(r[args.end_column]) < 10 or r[args.end_column].endswith("-01-01") or r[args.end_column].endswith("-31-31"):
                df.at[i, f"{args.end_column}_precision"] = "year"
            else:
                df.at[i, f"{args.end_column}_precision"] = "day"

    print(df, df.columns)
    df.to_csv(target_file, sep=args.sep, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-location", default=None)
    parser.add_argument("--meta-file", required=True, help="metadata file you want to add precision to")
    parser.add_argument("--sep", default=",")
    parser.add_argument("--start-column", default="start")
    parser.add_argument("--end-column", default="end")
    main(parser.parse_args())
