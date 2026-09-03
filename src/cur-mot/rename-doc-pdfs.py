#!/usr/bin/env python3
from tqdm import tqdm
import argparse, os, re




def main(args):
    pattern = re.compile(r'(?=\d)(?<=\D)(?=\D*?\d)')
    pattern2 = re.compile(r'(?<=\d)(?=\D)')
    yearpath=f"{args.doc_pdf_path}/{args.year}"
    files = os.listdir(f"{yearpath}")
    for f in tqdm(files):
        if "_" in f:
            continue
        print(f)
        spl = pattern.split(f)
        committee = spl[0]
        try:
            Next =  f"{int(spl[1][:-4]):0>4}.pdf"
        except:
            spl2 = pattern2.split(spl[1])
            Next = f"{int(spl2[0]):0>4}-{spl2[1]}"
        #print(f, "--> " f"mot_{args.year}_{committee}_{Next}")
        nn = f"mot_{args.year}_{committee}_{Next}"
        os.rename(f"{yearpath}/{f}", f"{yearpath}/{nn}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-pdf-path", default="riksdagen-motions-doc-pdf/data")
    parser.add_argument("-y", "--year", type=str, required=True)
    args = parser.parse_args()
    main(args)
