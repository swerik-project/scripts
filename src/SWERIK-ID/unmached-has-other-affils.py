#!/usr/bin/env python3
import argparse
import pandas as pd




def check(person_id, party_id, affils):
    only, more_party = None, None
    f = affils.loc[
        (affils["person_id"] == person_id) &
        (affils["party_id"] == party_id)
    ]
    if len(f) > 1:
        only, more_party = False, True
    else:
        f = affils.loc[affils["person_id"] == person_id]
        if len(f) == 1:
            only, more_party = True, False
        elif len(f) > 1:
            only, more_party = False, False
    return only, more_party




def main(args):
    affils = pd.read_csv("riksdagen-persons/data/party_affiliation.csv")
    df = pd.read_csv(args.infile, sep=args.sep)
    print("~~~~~~A:", affils.columns)
    print("------D:", df.columns)
    for i,r in df.iterrows():
        only_affil, others_same_p = check(r["person_id"], r["party_id"], affils)
        df.at[i, "persons_only_affiliation"] = only_affil
        df.at[i, "other_affil_same_party"] = others_same_p
    df.to_csv(args.infile, sep=args.sep, index=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infile", type=str, required=True, help="path to file")
    parser.add_argument("--sep", default=";")
    args = parser.parse_args()
    main(args)
