#!/usr/bin/env python3
from common.html_common import (
    add_to_docD,
    doc_D,
)
from glob import glob
from tqdm import tqdm
import argparse
import os
import pandas as pd


formatters = {
    "200506": {
        10.52: "body",                  # 264863
        17.71: "h1",                    # 3392
        8.86 : "fw-fn",                 # 3128
        14.94: "h2",                    # 2910
        14.39: "fw",                    # 1233
        22.14: "header_title",          # 916
        13.28: "body",                  # 786
        11.76: "emoji",                 # 317
        6.09: "footnote_ref",           # 264
        11.62: "h3",                    # 194
        9.41: "table_cell",             # 129
        6.42: "footnote_nr",            # 33
        19.93: "header_title",          # 28
        9.4: "skip",                    # 15
        8.81: "footnote_nr",            # 15
        5.09: "footnote_nr",            # 9
        11.07: "body",                  # 2
        11.05: "emoji",                 # 2
        9.9: "skip",                    # 2
        7.75: "footnote_nr-ref",        # 2
        7.08: "footnote_ref",           # 2
        18.23: "body",                  # 1
        12.45: "emoji",                 # 1
        12.18: "body",                  # 1
        "parts": {
            "header_block": ["h1", "header_author", "header_title"],
            "signature_block": ["body"],
        }
    }
}



def main(args):

    pdf_dumps = sorted(glob(f"riksdagen-motions-pdf/data/{args.parliament_year}/*/*.tsv"))
    tmp_counter = 0
    for d in tqdm(pdf_dumps):
        header_found = False
        current_div = None
        current_fragment = None
        last_formatter = None

        d_base = os.path.basename(d)
        print(d_base)

        df = pd.read_csv(d, sep='\t')
        doc_d = doc_D()
        for i, r in df.iterrows():

            if str(r["conf"]) == "100":
                formatter = formatters[args.parliament_year][r["height"]]
                #print("-", formatter==last_formatter, last_formatter, formatter)
                if formatter == "header_title":
                    header_found = True
                    #print(current_fragment)
                    if last_formatter == formatter:

                        current_fragment["text"].append(r["text"])
                    else:
                        if current_fragment is not None:
                            doc_d = add_to_docD(doc_d, current_fragment, current_div)
                        current_fragment = {}
                        current_div = "header_title"
                        if "text" not in current_fragment:
                            current_fragment["text"] = []
                        current_fragment["text"].append(r["text"])
                else:
                    if current_fragment is not None:
                        doc_d = add_to_docD(doc_d, current_fragment, current_div)
                    current_fragment = {}
                    current_div = "unknown"
                last_formatter = formatter


        if header_found:
            tmp_counter += 1
            print(" --", " ".join(doc_d["header_title"]["text"]))


    print(len(pdf_dumps), tmp_counter)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--parliament-year", required=True)
    args = parser.parse_args()
    main(args)
