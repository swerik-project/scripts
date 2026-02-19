#!/usr/bin/env python3
"""
Fix party affiliations that start too late --> split affiliation to two rows, add row with party successor.
"""
from tqdm import tqdm
import argparse
import pandas as pd
from pyriksdagen.utils import get_data_location



def get_suc_info(suc_id, party):
    suc_name, suc_wiki = None, None
    tgt = party.loc[party["swerik_party_id"] == suc_id]
    if len(tgt) == 1:
        i = list(tgt.index)[0]
        suc_name = party.at[i, "party"]
        suc_wiki= party.at[i, "party_id"]
    return suc_name, suc_wiki


def get_affil_idx(r, affil):
    tgt =  affil.loc[
        (affil["person_id"] == r["person"]) &
        (affil["start"] == r["affil_start"]) &
        (affil["end"] == r["affil_end"]) &
        (affil["swerik_party_id"] == r["party"])
        ]
    if len(tgt) == 1:
        return list(tgt.index)[0]
    elif len(tgt) == 0:
        tgt2 =  affil.loc[
            (affil["person_id"] == r["person"]) &
            (affil["start"] == r["affil_start"][:4]) &
            (affil["end"] == r["affil_end"][:4]) &
            (affil["swerik_party_id"] == r["party"])
            ]
        if len(tgt2) == 1:
            return list(tgt2.index)[0]
    raise Exception(f"target not == 1 :: {len(tgt)}, {len(tgt2)}")




def main(args):
    if args.metadata_folder is None:
        metadata_folder = get_data_location("metadata")
    else:
        metadata_folder = args.metadata_folder

    affil = pd.read_csv(f"{metadata_folder}/party_affiliation.csv")
    start_len = len(affil)
    print(start_len)
    party = pd.read_csv(f"{metadata_folder}/party.csv")
    df = pd.read_csv(args.infile, sep=args.sep)
    for i, r in tqdm(df.iterrows(), total=len(df)):
        print(i)
        affil_idx = get_affil_idx(r, affil)
        suc_name, suc_wiki = get_suc_info(r["suc"], party)
        if r["affil_start"] > r["party_end"]:
            affil.at[affil_idx, "party_id"] = suc_wiki
            affil.at[affil_idx, "party"] = suc_name
            affil.at[affil_idx, "swerik_party_id"] = r["suc"]
        else:
            affil.at[affil_idx, "end"] = r["party_end"]
            affil.loc[len(affil)] = {
                    "person_id": r["person"],
                    "start": r["party_end"],
                    "end": r["affil_end"],
                    "party": suc_name,
                    "party_id": suc_wiki,
                    "swerik_party_id": r["suc"]
                }
    affil.sort_values(by=["person_id", "start"], inplace=True)
    affil.to_csv(f"{metadata_folder}/party_affiliation.csv", index=False)
    print(start_len, "+", len(df), "==", len(affil), "   ::   ", start_len +len(df) == len(affil))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infile", required=True, type=str)
    parser.add_argument("--sep", default=";")
    parser.add_argument('--metadata_folder', type=str, default=None)
    args = parser.parse_args()
    main(args)
