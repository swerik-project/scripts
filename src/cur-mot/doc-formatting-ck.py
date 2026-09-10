#!/usr/bin/env python3
from glob import glob
from tqdm import tqdm
import argparse
import pandas as pd


def main(args):
    D = {}
    tsvs = sorted(glob(f"riksdagen-motions-pdf/data/{args.parliament_year}/*/*.tsv"))
    for tsv in tqdm(tsvs):
        df = pd.read_csv(tsv, sep='\t')
        print(df["conf"].unique())

        for i, r in df.iterrows():
            if r["conf"] == 100:
                if r["height"] not in D:
                    D[r["height"]] = 0
                D[r["height"]] += 1


    {print(k, ":", v) for k,v in dict(sorted(D.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)).items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-y", "--parliament-year", type = str, required=True)
    args = parser.parse_args()
    main(args)
