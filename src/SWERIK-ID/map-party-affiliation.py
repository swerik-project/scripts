#!/usr/bin/env python3
"""
Add a swerik party ID to the party_affiliation.csv file.
"""
from tqdm import tqdm
import pandas as pd
from pyriksdagen.utils import get_data_location
import argparse



def main(args):
    if args.metadata_folder is None:
        metadata_folder = get_data_location("metadata")
    else:
        metadata_folder = args.metadata_folder
    party_map = pd.read_csv(f"{metadata_folder}/party.csv")
    affiliation = pd.read_csv(f"{metadata_folder}/party_affiliation.csv")
    affiliation["swerik_party_id"] = None
    affiliation["hist_acc"] = None # historical accuracy

    onetoone = 0
    onetomany = 0
    onetomanyOK = 0


    for i, r in tqdm(affiliation.iterrows(), total=len(affiliation)):
        filt = party_map.loc[party_map['party_id'] == r['party_id']]
        hist_acc = True
        s = r["start"]
        e = r["end"]
        if len(filt) == 1:
            onetoone += 1
            affiliation.at[i, 'swerik_party_id'] = filt.at[filt.loc[filt['party_id'] == r['party_id']].index.to_list()[0], 'swerik_party_id']
            affiliation.at[i, 'party'] = filt.at[filt.loc[filt['party_id'] == r['party_id']].index.to_list()[0], 'party']
            #print("~", e, pd.isnull(e), "|", filt.at[filt.loc[filt['party_id'] == r['party_id']].index, "dissolution"])
            if pd.notnull(s):
                if len(s) == 4:
                    if pd.notnull(s) and s < filt.at[filt.loc[filt['party_id'] == r['party_id']].index.to_list()[0],'inception'][:4]:
                        print(s, filt.at[filt.loc[filt['party_id'] == r['party_id']].index.to_list()[0],'inception'][:4])
                        hist_acc = False
                elif len(s) > 4:
                    if pd.notnull(s) and s < filt.at[filt.loc[filt['party_id'] == r['party_id']].index.to_list()[0],'inception']:
                        print(s, filt.at[filt.loc[filt['party_id'] == r['party_id']].index.to_list()[0],'inception'])
                        hist_acc = False

            elif pd.notnull(filt.at[filt.loc[filt['party_id'] == r['party_id']].index.to_list()[0], 'dissolution']):
                if pd.notnull(e) and e > filt.at[filt.loc[filt['party_id'] == r['party_id']].index.to_list()[0], 'dissolution']:
                    hist_acc = False

        else:
            onetomany += 1
            if pd.notnull(s):
                filt = filt.loc[filt["inception"] <= s]
            if pd.notnull(e):
                filt = filt.loc[(pd.isnull(filt['dissolution'])) | (filt['dissolution'] >= e)]
            if len(filt) == 1:
                affiliation.at[i, 'swerik_party_id'] = filt.at[filt.loc[filt['party_id'] == r['party_id']].index.to_list()[0], 'swerik_party_id']
                affiliation.at[i, 'party'] = filt.at[filt.loc[filt['party_id'] == r['party_id']].index.to_list()[0], 'party']
            else:
                hist_acc = False

        affiliation.at[i, 'hist_acc'] = hist_acc

    print(affiliation)
    print(len(affiliation.loc[pd.notnull(affiliation["swerik_party_id"])]))
    print(len(affiliation.loc[pd.notnull(affiliation["hist_acc"])]), len(affiliation.loc[affiliation['hist_acc'] == True]))
    print(affiliation.loc[(affiliation['hist_acc'] == False)|(pd.isna(affiliation['hist_acc']))])

    # TODO: mkdir etc. for this to work
    #affiliation.loc[(affiliation['hist_acc'] == False)|(pd.isna(affiliation['hist_acc']))].to_csv(f"{metadata_folder}/test/result/party-problems.csv", index=False)
    
    print(onetoone, onetomany)
    print(len(affiliation.loc[pd.isnull(affiliation['start'])]))

    affiliation.drop(columns=['hist_acc'], inplace=True)
    affiliation.to_csv(f"{metadata_folder}/party_affiliation.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata_folder', type=str, default=None)
    args = parser.parse_args()
    main(args)
