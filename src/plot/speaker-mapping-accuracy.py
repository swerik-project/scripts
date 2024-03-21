#!/usr/bin/env python3
"""
Draw a graph of the introduction mapping accuracy estimate (current-version only)
"""
from multiprocessing import Pool
from pyriksdagen.utils import (
    get_data_location,
    parse_protocol,
    protocol_iterators,
)
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def plot(df):
    df = df.reset_index().rename(columns={"index":"year"})
    df = df[['year', 'accuracy']]
    colors = list('bgrcmyk')
    plt.rc('axes')
    f, ax = plt.subplots()
    x = df['year'].tolist()
    y = df['accuracy'].tolist()
    x, y = zip(*sorted(zip(x,y),key=lambda x: x[0]))
    plt.plot(x, y, linewidth=1.75, label='_nolegengd_')
    plt.rcParams.update({'font.size': 14})
    plt.axhline(y=0.90, color='green', linestyle='--', linewidth=1, label='_nolegend_')
    plt.title('Estimated accuracy for identification of speaker')
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return f, ax


def get_date(root, ns):
    for docDate in root.findall(f".//{ns['tei_ns']}docDate"):
        date_string = docDate.text
        break
    return date_string


def accuracy(protocol):
    root, ns = parse_protocol(protocol, get_ns=True)
    year = int(get_date(root, ns).split("-")[0])
    known, unknown = 0, 0
    for div in root.findall(f".//{ns['tei_ns']}div"):
        for elem in div:
            if "who" in elem.attrib:
                who = elem.attrib["who"]
                if who == "unknown":
                    unknown += 1
                else:
                    known += 1
    return year, known, unknown




def main(args):
    protocols = sorted(list(protocol_iterators(get_data_location('records'))))
    if args.start is not None:
        protocols = sorted(list(protocol_iterators(
                                        get_data_location('records'),
                                        start=args.start,
                                        end=args.end)))
    years = sorted(set([int(p.split('/')[2][:4]) for p in protocols]))
    years.append(max(years)+1)
    df = pd.DataFrame(
        np.zeros((len(years), 2), dtype=int),
        index=years, columns=['known', 'unknown'])
    pool = Pool()
    for year, known, unknown in tqdm(pool.imap(accuracy, protocols), total=len(protocols)):
        df.loc[year, 'known'] += known
        df.loc[year, 'unknown'] += unknown
    df['accuracy'] = df.div(df.sum(axis=1), axis=0)['known']
    #print(df)
    print("Average:", df['accuracy'].mean())
    print("Weighted average:", df["known"].sum() / (df["known"] + df["unknown"]).sum())
    print("Minimum: {} ({})".format(*[getattr(df['accuracy'], f)() for f in ['min', 'idxmin']]))


    f, ax = plot(df)
    plt.savefig(f'input/plots/LREC/speaker_mapping_accuracy.pdf',
                dpi=300, format='pdf')
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--start", type=int, default=None)
    parser.add_argument("-e", "--end", type=int, default=None)
    args = parser.parse_args()
    main(args)

