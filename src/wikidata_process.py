'''
Process metadata from corpus/metadata into easy-to-use tables, and save them in input/
Necessary for redetect.py and other scripts that rely on metadata.
'''
from pyriksdagen.metadata import load_Corpus_metadata
import argparse

def main(args):

    corpus = load_Corpus_metadata()
    for file in ['member_of_parliament', 'minister', 'speaker']:
        df  = corpus[corpus['source'] == file]
        
        # Sort the df to make easier for git
        sortcols = list(df.columns)
        print(f"sort by {sortcols}")
        df = df.sort_values(sortcols)
        df.to_csv(f"{args.outfolder}/{file}.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outfolder", type=str, default="input/matching")
    args = parser.parse_args()

    main(args)
