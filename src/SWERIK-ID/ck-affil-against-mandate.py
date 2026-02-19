#!/usr/bin/env python3

import argparse
import pandas as pd




def get_party_se():
    d = {}
    df = pd.read_csv("riksdagen-persons/data/party.csv")
    for i, r in df.iterrows():
        d[r['party']] = {
            "swerik_party_id": r["swerik_party_id"],
            "start": r["inception"],
            "start_p": r["inception_precision"],
            "end": r["dissolution"],
            "end_p": r["dissolution_precision"]
        }
    return d



def get_mandate_se(person_id, mandates):
    earliest, latest = None, None
    pm = mandates.loc[mandates["person_id"]==person_id]
    earliest = pm['start'].min()
    try:
        latest = pm['end'].max()
    except:
        pass
    return earliest, latest




def main(args):
    parties_se = get_party_se()
    mandates = pd.read_csv("riksdagen-persons/data/member_of_parliament.csv")
    df = pd.read_csv(args.infile, sep=args.sep)
    for i,r in df.iterrows():
        if pd.isnull(r["start"]) and pd.isnull(r["end"]):
            mandate_s, mandate_e = get_mandate_se(r["person_id"], mandates)
            if pd.isnull(mandate_s):
                continue
            print(mandate_s, mandate_e)
            df.at[i, "mandate_lower"] = mandate_s
            df.at[i, "mandate_upper"] = mandate_e
            try:
                party_se = parties_se[r["party"]]
                print(party_se)
            except:
                print(f"no such party : {r['party']}")
            if len(mandate_s) > 4 and party_se["start_p"] == "year":
                start_in_range = mandate_s >= party_se["start"]
            else:
                start_in_range = mandate_s[:4] >= party_se["start"][:4]
            if pd.notnull(party_se["end"]):
                if pd.notnull(mandate_e) and mandate_e is not None:
                    if len(mandate_e) > 4 and party_se["end_p"] == "year":
                        end_in_range = mandate_e >= party_se["end"]
                    else:
                        end_in_range = mandate_e[:4] >= party_se["end"][:4]
                else:
                    end_in_range=True
            else:
                end_in_range = True

            if start_in_range == end_in_range:
                if start_in_range == True:
                    df.at[i, "in_range"] = "yes"
                    df.at[i, "swerik_party_id"] = party_se["swerik_party_id"]
                else:
                    if mandate_s > party_se["end"] or mandate_s < party_se['start']:
                        df.at[i, "in_range"] = "no_overlap"
                    else:
                        df.at[i, "in_range"] = "mandate_contains_affil"
            else:
                if start_in_range == False:
                    df.at[i, "in_range"] = "affil_too_early"
                else:
                    df.at[i, "in_range"] = "affil_too_late"

    df.to_csv(args.infile, sep=args.sep, index=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infile", type=str, required=True, help="path to file")
    parser.add_argument("--sep", default=";")
    args = parser.parse_args()
    main(args)
