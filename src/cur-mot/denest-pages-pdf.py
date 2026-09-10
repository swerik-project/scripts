#!/usr/bin/env python3
"""
Denest pages/ directory in the pdf repository.
"""
import argparse
import os
import shutil
from glob import glob

def main(args):
    pdf_packages = sorted(glob(f"riksdagen-motions-pdf/data/{args.parliament_year}/*/"))
    for p in pdf_packages:
        p_content = glob(f"{p}pages/*")
        for file_ in p_content:
            print(" --", file_)
            if file_.endswith(".png"):
                if args.remove_png == True:
                    os.remove(file_)
                    print("rm:", file_)
                else:
                    shutil.move(file_, p)
                    print("mv:", file_, p)
            elif file_.endswith(".pdf"):
                shutil.move(file_, p)
                print("mv:", file_, p)
            else:
                os.remove(file_)
                print("rm:", file_)
        os.rmdir(f"{p}pages/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-y", "--parliament-year", type = str, required=True)
    parser.add_argument("-P", "--remove-png", type=bool, default=True)
    args = parser.parse_args()
    main(args)
