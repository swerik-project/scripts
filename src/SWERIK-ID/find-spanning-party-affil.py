#!/usr/bin/env python3
"""
Add a row for party affiliations that occur before formation / name change of a party or after dissolution / name change.
"""
from tqdm import tqdm
import argparse
import numpy as np
import os
import pandas as pd
import warnings



def find_predecessor(party_id, party):
    f = party.loc[party["swerik_successor"].str.contains(party_id, na=False)]
    if len(f) > 0:
        pred_id = list(f["swerik_party_id"])
        pred_name = list(f["party"])
        pred_start = list(f["inception"])
        pred_end = list(f["dissolution"])
        return pred_id, pred_name, pred_start, pred_end
    else:
        return None, None, None, None


def main(args):

    party = pd.read_csv(args.parties)
    affil = pd.read_csv(args.affiliations)
    print(party, affil)
    print(affil.columns)
    print(party.columns)

    D = {}
    s_count = 0
    e_count = 0
    rows = []
    cols = ["person", "party", "party_name", "problem", "party_start", "psp", "affil_start", "asp", "party_end", "pep", "affil_end", "pap", "sev"]
    for i, r in tqdm(affil.iterrows(), total = len(affil)):
        s = r["start"]
        if len(str(s)) == 4:
            s = f"{s}-01-01"
        sp = r["start_precision"]
        e = r["end"]
        if len(str(e)) == 4:
            e = f"{e}-12-31"
        ep = r["end_precision"]
        p = r["swerik_party_id"]

        person = r["person_id"]

        if p in D:
            ps = D[p]["start"]
            psp = D[p]["psp"]
            pe = D[p]["end"]
            pep = D[p]["pep"]
            pre_p = D[p]["predecessor"]
            post_p = D[p]["successor"]

        else:
            try:
                p_i = party.loc[party['swerik_party_id'] == p].index[0]
            except:
                p_i = None
            #    print("~~~", p)
            #print(p_i)

            if p_i:
                ps = party.at[p_i, "inception"]
                psp = party.at[p_i, "inception_precision"]
                pe = party.at[p_i, "dissolution"]
                pep = party.at[p_i, "dissolution_precision"]
                if pd.notnull(pe) and pe.endswith("-01-01"):
                    pe = f"{pe[:4]}-12-31"
                post_p = party.at[p_i, "successor_id"]
                if pd.notnull(post_p):
                    post_p = post_p.split("|")
                else:
                    post_p = []

                party_name = party.at[p_i, "party"]



                if pd.notnull(ps) and pd.notnull(s):
                    start_error = None
                    if sp != "day" or psp != "day":
                        if s[:4] < ps[:4:]:
                            start_error = True
                    else:
                        if s < ps:
                            start_error = True
                    if start_error:
                        pred_id, pred_name, pred_start, pred_end = find_predecessor(p, party)

                        if pred_id is None:
                            print(f"XXX XXX     No predecessor id found for {p}")
                        elif len(pred_id) == 1 and \
                            len(pred_id) == len(pred_name) and \
                            len(pred_name)== len(pred_start) and \
                            len(pred_start)== len(pred_end):
                            print(f"    ~~ predecessor found {pred_id} / {pred_name} succeded by {p} / {party_name}")
                        else:
                            print(f"XXX xxx    More than one predecessor found, or some other whacky problem found for {p}")

                        if s[:4] == ps[:4]:
                            rows.append([person, p, party_name, "affil starts too early", ps, s, pe, e, 2])
                        else:
                            print(f"!!! -- party {p} didn't exist until {ps}.. affil starts on {s}")
                            rows.append([person, p, party_name, "affil starts too early", ps, psp, s, sp, pe, pep, e, ep, 1])
                            s_count += 1

                if pd.notnull(pe) and pd.notnull(e):
                    end_error = None
                    if ep != "day" or pep != "day":
                        if e[:4] > pe[:4]:
                            end_error = True
                    else:
                        if e > pe:
                            end_error = True
                    if end_error:
                        if e.endswith("-12-31") and pe.endswith("-01-01"):
                            rows.append([person, p, party_name, "affil ends too late", ps, psp, s, sp, pe, pep, e, ep, 2])
                        else:
                            print(f"!!! ++ party {p} ceased to exist on {pe}.. affil ends on {e}")
                            e_count += 1
                            rows.append([person, p, party_name, "affil ends too late", ps, psp, s, sp, pe, pep, e, ep, 1])

    print(s_count, e_count)
    df = pd.DataFrame(rows, columns=cols)
    df.sort_values(by=["sev", "party"], inplace=True)
    increment = 1
    while os.path.exists(f"riksdagen-persons/test/result/_party-problem-{increment}.csv"):
        increment += 1
    df.to_csv(f"riksdagen-persons/test/result/_party-problem-{increment}.csv", sep=";")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parties", default="riksdagen-persons/data/party.csv")
    parser.add_argument("--affiliations", default="riksdagen-persons/data/party_affiliation.csv")
    args = parser.parse_args()
    main(args)
