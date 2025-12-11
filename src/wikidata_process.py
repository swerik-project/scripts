'''
Process metadata from corpus/metadata into easy-to-use tables, and save them in input/
Necessary for redetect.py and other scripts that rely on metadata.
'''
from pyriksdagen.metadata import load_Corpus_metadata
import argparse
import pandas as pd
import sqlite3
from pathlib import Path

def main(args):

    if not Path(outfolder).exists():
        raise ValueError(f"output folder ({outfolder}) does not exist. Exiting...")

    corpus = load_Corpus_metadata()

    if "xlsx" in args.formats:
        excel_writer = pd.ExcelWriter(args.outfile_excel, engine='xlsxwriter')
    if "sqlite" in args.formats:
        conn = sqlite3.connect(args.outfile_sqlite)

    for file in ['member_of_parliament', 'minister', 'speaker']:
        df  = corpus[corpus['source'] == file]
        
        # Sort the df to make easier for git
        sortcols = list(df.columns)
        print(f"sort by {sortcols}")
        df = df.sort_values(sortcols)
        if "csv" in args.formats:
            df.to_csv(f"{args.outfolder}/{file}.csv", index=False)

        table_name = f"processed_{file}"
        if "xlsx" in args.formats:
            df.to_excel(excel_writer, sheet_name=table_name, index=False)
        if "sqlite" in args.formats:
            df.to_sql(table_name, conn, if_exists='replace', index=False)

    if "sqlite" in args.formats:
        conn.close()

    if "xlsx" in args.formats:
        excel_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outfolder", type=str, default="input/matching")
    parser.add_argument("--formats", type=str, nargs="+", default=["csv", "sqlite", "xlsx"])
    parser.add_argument("--outfile_sqlite", default="RiksdagenPersons.db")
    parser.add_argument("--outfile_excel", default="RiksdagenPersons.xlsx")
    args = parser.parse_args()

    main(args)
