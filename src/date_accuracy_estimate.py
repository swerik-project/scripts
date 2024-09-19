"""
Calculate an upper bound for segment classification accuracy.
Based on the gold standard annotations.
"""
from pyriksdagen.utils import protocol_iterators, elem_iter, infer_metadata
from lxml import etree
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from multiprocessing import Pool
from pathlib import Path
import warnings
import progressbar
from scipy.stats import beta
import seaborn as sns
from matplotlib import pyplot as plt
from pyriksdagen.utils import TEI_NS
from datetime import datetime

XML_NS = "{http://www.w3.org/XML/1998/namespace}"


# Fix parallellization
def date_range_from_protocol(protocol):
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.parse(protocol, parser).getroot()
    mindate, maxdate = None, None
    for docDate in root.findall(f".//{TEI_NS}docDate"):
        if mindate is None or mindate > docDate.get("when"):
            mindate = docDate.get("when")
        if maxdate is None or maxdate < docDate.get("when"):
            maxdate = docDate.get("when")
    return mindate, maxdate

def get_jaccard(startdate, enddate, startdate_hat, enddate_hat):
    startdate = datetime.fromisoformat(f'{startdate} 00:00:01')
    startdate_hat = datetime.fromisoformat(f'{startdate_hat} 00:00:01')
    enddate = datetime.fromisoformat(f'{enddate} 23:59:59')
    enddate_hat = datetime.fromisoformat(f'{enddate_hat} 23:59:59')

    union = min(startdate, startdate_hat), max(enddate, enddate_hat) 
    intersection = max(startdate, startdate_hat), min(enddate, enddate_hat) 
    if intersection[1] <= intersection[0]:
        return 0.0, 0.0, 0.0, 0.0

    unionlen = (union[1] - union[0]).total_seconds()
    intersectionlen = (intersection[1] - intersection[0]).total_seconds()
    contain = int(startdate >= startdate_hat and enddate <= enddate_hat)
    return intersectionlen / unionlen, int(intersectionlen == unionlen), int(intersectionlen / unionlen > 0.0), contain

def main(args):
    protocols = list(protocol_iterators(args.records_folder, start=args.start, end=args.end))
    print(args.path_goldstandard)
    df = pd.read_csv(args.path_goldstandard)
    print(df)
    df = df[df["path"].notnull()]
    df = df[df["path"].str.contains("/")]
    df["protocol_id"] = df["path"].str.split("/").str[-1].str.split(".").str[0]
    print(df)
    rows = []
    correct, incorrect = 0, 0
    jaccs, perfects, overlaps, contains = [], [], [], []
    zero_overlaps = []
    for p in progressbar.progressbar(protocols):
        path = Path(p)
        protocol_id = path.stem
        df_p = df[df["protocol_id"] == protocol_id]
        if len(df_p) == 1:
            startdate, enddate = None, None
            datestr = list(df_p["true-dates"])[0]
            startdate_hat, enddate_hat = date_range_from_protocol(path)
            try:
                if " - " in datestr:
                    startdate, enddate = datestr.split(" - ")
                else:
                    startdate, enddate = datestr, datestr

                jacc, perfect, overlap, contain = get_jaccard(startdate, enddate, startdate_hat, enddate_hat)
                jaccs.append(jacc)
                perfects.append(perfect)
                overlaps.append(overlap)
                contains.append(contain)
                if overlap == 0:
                    print()
                    print(protocol_id)
                    print("true:"  +startdate + " - " + enddate, startdate_hat + " - " + enddate_hat)
                    zero_overlaps.append(protocol_id)

            except:
                print("Problem with", protocol_id, datestr)
            #print(jacc)

        elif len(df_p) > 1:
            print("Problem with", protocol_id)
        
    print("E[J]", np.mean(jaccs))
    print("P(J == 1)", np.mean(perfects))
    print("P(J > 0)", np.mean(overlaps))
    print("Contains", np.mean(contains))

    zero_overlaps = "\n".join(zero_overlaps)
    print(f"Zero overlap in: {zero_overlaps}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1867)
    parser.add_argument("--end", type=int, default=2022)
    parser.add_argument("--records_folder", type=str, default="corpus/records")
    parser.add_argument("--path_goldstandard", type=str, default="date-sample.csv")
    args = parser.parse_args()
    df = main(args)

    print(df)
