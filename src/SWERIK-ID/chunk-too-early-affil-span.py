#!/usr/bin/env python3
"""
Fix party affiliations that start too early --> split affiliation to two rows, add row with party predecessor.
"""
from tqdm import tqdm
import argparse
import pandas as pd
import warnings
from pyriksdagen.utils import get_data_location



def get_pred_info(pred_id, party):
    pred_name, pred_wiki = None, None
    tgt = party.loc[party["swerik_party_id"] == pred_id]
    if len(tgt) == 1:
        i = list(tgt.index)[0]
        pred_name = party.at[i, "party"]
        pred_wiki= party.at[i, "party_id"]
    return pred_name, pred_wiki


def get_affil_idx(r, affil):
    try:
        tgt =  affil.loc[
            (affil["person_id"] == r["person"]) &
            (affil["start"] == r["affil_start"]) &
            (affil["end"] == r["affil_end"]) &
            (affil["swerik_party_id"] == r["party"])
            ]
        assert len(tgt) == 1
    except:
        tgt =  affil.loc[
            (affil["person_id"] == r["person"]) &
            (affil["start"] == r["affil_start"]) &
            (affil["end"] == r["affil_end"])
            ]
    if len(tgt) == 1:
        return list(tgt.index)[0]

    else:
        if len(tgt) == 0:
            tgt = affil.loc[
            (affil["person_id"] == r["person"]) &
            (affil["party_id"] == r["party_id"])
        ]
        if len(tgt) == 1:
            return list(tgt.index)[0]

    #print(r)
    print(tgt)
    #raise ValueError("target not == 1")
    return None



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
    wc = 0
    for i, r in tqdm(df.iterrows(), total=len(df)):
        print(i)
        affil_idx = get_affil_idx(r, affil)
        if affil_idx is not None:
            pred_name, pred_wiki = get_pred_info(r["pred"], party)
            if r["affil_end"] < r["party_start"]:
                affil.at[affil_idx, "party_id"] = pred_wiki
                affil.at[affil_idx, "party"] = pred_name
                affil.at[affil_idx, "swerik_party_id"] = r["pred"]
            else:
                affil.at[affil_idx, "start"] = r["party_start"]
                affil.at[affil_idx, "swerik_party_id"] = r["swerik_party_id"]
                if pd.isna(affil.at[affil_idx, "end"]):
                    affil.at[affil_idx, "end"] = r["affil_end"]
                affil.loc[len(affil)] = {
                        "person_id": r["person"],
                        "start": r["affil_start"],
                        "end": r["party_start"],
                        "party": pred_name,
                        "party_id": pred_wiki,
                        "swerik_party_id": r["pred"]
                    }
        else:
            warnings.warn("The provided party affiliation was not found.")
            wc += 1
            print(r)
    affil.sort_values(by=["person_id", "start"], inplace=True)
    affil.to_csv(f"{metadata_folder}/party_affiliation.csv", index=False)
    print(start_len, "+", len(df), "==", len(affil), "   ::   ", start_len +len(df) == len(affil))

    print("\n\n\n", wc, "\n\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infile", required=True, type=str)
    parser.add_argument("--sep", default=";")
    parser.add_argument('--metadata_folder', type=str, default=None)
    args = parser.parse_args()
    main(args)
