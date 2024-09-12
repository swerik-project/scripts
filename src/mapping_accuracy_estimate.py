"""
Calculate an upper bound for introduction mapping accuracy
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
import pandas as pd




def get_date(root, ns):
    for docDate in root.findall(f".//{ns['tei_ns']}docDate"):
        date_string = docDate.text
        break
    return date_string


# Fix parallellization
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
    df['accuracy_upper_bound'] = df.div(df.sum(axis=1), axis=0)['known']
    print(df)
    print("Average:", df['accuracy_upper_bound'].mean())
    print("Weighted average:", df["known"].sum() / (df["known"] + df["unknown"]).sum())
    print("Minimum: {} ({})".format(*[getattr(df['accuracy_upper_bound'], f)() for f in ['min', 'idxmin']]))
    df.to_csv("input/accuracy/upper_bound.csv", index_label='year')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--start", type=int, default=None)
    parser.add_argument("-e", "--end", type=int, default=None)
    args = parser.parse_args()
    df = main(args)

