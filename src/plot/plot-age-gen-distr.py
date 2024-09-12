#!/usr/bin/env python3
"""
Plot age and gender distribution of MPs
"""
from pyriksdagen.date_handling import yearize_mandates
from pyriksdagen.metadata import load_Corpus_metadata
from pyriksdagen.utils import get_data_location
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import subprocess




def mk_year_d():
    D = {}
    df = yearize_mandates(metadata_folder=get_data_location('metadata'))
    mps = df.loc[df['role'].str.endswith('ledamot')].copy()
    pys = sorted(mps['parliament_year'].unique(), key = lambda x: str(x)[:4])
    print(pys)
    print(mps['role'].unique())

    for py in pys:
        D[py] = list(mps.loc[mps['parliament_year'] == py, 'person_id'].unique())
    return D


def mk_year_dfs(D, person):
    gen_rows = []
    gen_cols = ["year", "male", "female", "unspec"]
    age_rows = []
    age_cols = ["year", "age"]
    for k, v in D.items():
        if int(str(k)[:4]) < 2023:
            male = 0
            female = 0
            unspec = 0
            ages = []
            for person_id in v:
                fpers = person.loc[person['person_id'] == person_id].copy()
                if len(fpers) > 0:
                    genders = fpers['gender'].unique()
                    if 'man' in genders and 'woman' in genders:
                        print("trans?")
                        unspec += 1
                    elif 'man' in genders:
                        male += 1
                    elif 'woman' in genders:
                        female += 1
                    else:
                        unspec += 1
                    try:
                        dob = fpers.loc[pd.notnull(fpers['born']), 'born'].unique()
                    except:
                        dob = None
                        print(person_id, "No DOB")
                    else:
                        if len(dob) > 0:
                            xyz = int(str(k)[:4])-int(str(dob[0])[:4])
                            if xyz > 10:
                                age_rows.append([k, xyz])
                            else:
                                print(person_id, "too young?", xyz)
                else:
                    print(f"ERRRMAGERD, no person {wiki_id} in person.csv")
            gen_rows.append([k, male, female, unspec])

    age_df = pd.DataFrame(age_rows, columns=age_cols)
    gen_df = pd.DataFrame(gen_rows, columns=gen_cols)

    return age_df, gen_df


def gender_preprocess(gender_df):
    def _div(a,b):
        return a/b
    def _add(a,b,c):
        for _ in [a,b,c]:
            if pd.isnull(_):
                _=0
        return a+b+c
    gender_df['year'] = gender_df['year'].apply(lambda x: str(x)[:4])
    gender_df['total'] = gender_df.apply(lambda x: _add(x['male'], x['female'], x['unspec']), axis=1)
    gender_df['male_p'] = gender_df.apply(lambda x: _div(x['male'], x['total']), axis=1)
    gender_df['female_p'] = gender_df.apply(lambda x: _div(x['female'], x['total']), axis=1)
    gender_df['unspec_p'] = gender_df.apply(lambda x: _div(x['unspec'], x['total']), axis=1)
    gender_df.drop(['male', 'female', 'unspec', 'unspec_p', 'total'], axis=1, inplace=True) # no unspec gender was visible in plot
    gender_df.set_index('year', inplace=True)
    return gender_df


def plot_gender_line2(gender_df, out):
    p, a = plt.subplots()
    plt.rcParams.update({'font.size': 14})
    a.plot(gender_df['female_p'])
    a.set_ylim(0, 1)
    plt.axhline(y=0.5, color='green', linestyle='--', linewidth=1, label='_nolegend_')
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    #start, end = a.get_xlim()
    #a.xaxis.set_ticks(np.arange(start, end, 20))
    ax = p.gca().xaxis
    ticks = ax.get_ticklines()
    [_.set_visible(False) for _ in ticks]
    for i, _ in enumerate(ax.get_ticklabels()):
        if int(_.get_text()) % 20 != 0:
            _.set_visible(False)
            #ticks[i].set_visible(False)
    #p.suptitle("Proportion of female members of parliament")
    p.set_size_inches(7,6)
    p.tight_layout()
    plt.savefig(f"{out}/prop-female2.pdf", format='pdf', dpi=300)
    plt.show()



def main(args):
    if args.metadata_path:
        metadata_path = args.metadata_path
    else:
        metadata_path = get_data_location('metadata')

    person = pd.read_csv(f"{metadata_path}/person.csv")
    D = mk_year_d()

    age_df, gen_df = mk_year_dfs(D, person)
    plot_gender_line2(gender_preprocess(gen_df), args.output_path)

    age_df["year"] = age_df['year'].apply(lambda x: str(x)[:4])
    age_df.to_csv(f"{args.output_path}/ages.csv")
    p = subprocess.Popen("scripts/src/plot/age-ribbon.r")
    p.wait()
    os.remove(f"{args.output_path}/ages.csv")

    print(len(person), len(D))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="input/plots/LREC")
    args = parser.parse_args()
    main(args)
