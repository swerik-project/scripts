#!/usr/bin/env python3
"""
pdftotext on
"""
import argparse
import os
import subprocess
from glob import glob
from tqdm import tqdm


def main(args):
    pdfs = sorted(glob(f"riksdagen-motions-pdf/data/{args.parliament_year}/*.pdf"))
    for pdf in tqdm(pdfs):
        d = os.path.dirname(pdf)
        b = os.path.basename(pdf)[:-4]
        subprocess.run([
                "pdftotext", "-tsv", pdf,
                f"{d}/{b}/{b}.tsv"
            ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-y", "--parliament-year", type = str, required=True)
    args = parser.parse_args()
    main(args)
