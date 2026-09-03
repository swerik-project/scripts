#!/usr/bin/env python3
from common.xml_utils import (
    parse_xml,
    write_xml,
)
from datetime import datetime
from lxml import etree
from tqdm import tqdm
import argparse, os
import pandas as pd


def main(args):
    orcid = os.environ.get("ORCID")
    today = datetime.now().strftime('%Y-%m-%d')
    df = pd.read_csv(f"{args.infile}", sep=';')
    mots = df["file"].unique()
    for mot in tqdm(mots):
        root, ns = parse_xml(f"riksdagen-motions/{mot}")
        print(root, ns)
        try:
            revdesc = root.find(f".//{ns['tei_ns']}revisionDesc")
            assert revdesc is not None
        except:
            revdesc = etree.SubElement(root.find(f"{ns['tei_ns']}teiHeader"), "revisionDesc")

        mot_df = df.loc[df["file"] == mot]
        for i, r in mot_df.iterrows():
            c = etree.SubElement(revdesc, "correction")
            c.text = "OCR correction"
            c.attrib["who"] = orcid
            c.attrib["when"] = today
            c.attrib["corresp"] = r['elem']
        write_xml(root, f"riksdagen-motions/{mot}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default="riksdagen-motions/_tmp.txt")
    args = parser.parse_args()
    main(args)
