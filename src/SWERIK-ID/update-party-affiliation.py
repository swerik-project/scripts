#!/usr/bin/env python3
"""
Take a csv file with updated party affiliations (correct historical inaccuracies) and update the party_affiliations.csv file.
"""
import argparse
import pandas as pd
from pyriksdagen.utils import get_data_location



def main(args):
    if args.metadata_folder is None:
        metadata_folder = get_data_location("metadata")
    else:
        metadata_folder = args.metadata_folder

    affiliation = pd.read_csv(f"{metadata_folder}/party_affiliation.csv")
    print("affil:", affiliation.columns)
    df = pd.read_csv(args.infile, sep=args.sep)
    print("corrections:", df.columns)
    for i, r in df.iterrows():
        #print(r["end"], type(r["end"]))
        f = affiliation.loc[
            (affiliation["person_id"] == r["person_id"])
            &
            (((pd.isnull(affiliation['start'])) & (pd.isnull(r["start"])))|(affiliation["start"] == r["start"]))
            &
            (((pd.isnull(affiliation['end'])) & (pd.isnull(r["end"])))|(affiliation["end"] == r["end"]))
            &
            (affiliation["party_id"] == r["party_id"])
        ]
        if len(f) != 1:
            print(len(f), r["start"], r["end"])
        else:
            #print(f.index.to_list())
            affiliation.at[f.index.to_list()[0], "swerik_party_id"] = r["swerik_id_correction"]
            affiliation.at[f.index.to_list()[0], "party"] = r["party_name_correction"]

    affiliation.sort_values(by=["person_id", "start"], inplace=True)
    affiliation.drop_duplicates(inplace=True)
    print(affiliation)
    affiliation.to_csv(f"{metadata_folder}/party_affiliation.csv", index=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infile", type=str, required=True, help="path to file")
    parser.add_argument("--sep", default=";")
    parser.add_argument('--metadata_folder', type=str, default=None)
    args = parser.parse_args()
    main(args)
