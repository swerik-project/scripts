#!/usr/bin/env python3
"""
Get the intro text and add it to the unknowns file
"""
from tqdm import tqdm
from pyriksdagen.segmentation import intro_to_dict
from pyriksdagen.utils import parse_tei
import argparse
import pandas as pd

def main(args):
    df = pd.read_csv(args.unknowns_file)
    df["intro_text"] = None
    df["intro_dict"] = None
    protocols = df["protocol_id"].unique()
    for protocol in tqdm(protocols):
        pdf = df.loc[df["protocol_id"] == protocol]
        root, ns = parse_tei(f"riksdagen-records/data/{protocol.split('-')[1]}/{protocol}")
        for i, r in pdf.iterrows():
            intro = root.find(f".//{ns['tei_ns']}note[@{ns['xml_ns']}id=\"{r['uuid']}\"]")
            if intro is not None and intro.text.strip() != '':
                t = ' '.join([_.strip() for _ in intro.text.splitlines() if _.strip() != ""])
                df.at[i, "intro_text"] = t
                df.at[i, "intro_dict"] = intro_to_dict(t)
            else:
                raise ValueError(f"intro not found :: {protocol}, {r['uuid']}")

    df.to_csv(args.outfile, sep=';', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-u", "--unknowns-file", default="input/matching/unknowns.csv")
    parser.add_argument("-o", "--outfile", default="input/matching/unknowns+intro_text.csv")
    args = parser.parse_args()
    main(args)

