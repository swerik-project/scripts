#!/usr/bin/env python3

import argparse
import os
import pandas as pd

def main(args):
    df = pd.read_csv(args.affiliations)
    print(len(df), len(df.loc[pd.isna(df["swerik_party_id"])]))

    increment = 1
    while os.path.exists(f"riksdagen-persons/test/result/_party-no-swerik-id-{increment}.csv"):
        increment += 1
    df.loc[pd.isna(df["swerik_party_id"])].to_csv(f"riksdagen-persons/test/result/_party-no-swerik-id-{increment}.csv", sep=";")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parties", default="riksdagen-persons/data/party.csv")
    parser.add_argument("--affiliations", default="riksdagen-persons/data/party_affiliation.csv")
    args = parser.parse_args()
    main(args)
