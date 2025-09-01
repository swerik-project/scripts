#!/usr/bin/env python3
from alto import parse_file, String
from glob import glob
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from tqdm import tqdm
import argparse, os, shutil


def main(args):

    for mot in tqdm(args.motions):
        if "fört" in mot:
            b = os.path.basename(mot)
            b = b.replace("0fört", "fört")
            shutil.move(mot, f"{args.motionspath}/fort/{b}")
            print("moved:", b)



if __name__ == '__main__':
    parser = fetch_parser("motions")
    main(impute_args(parser.parse_args()))
