#!/usr/bin/env python3
"""
Convert the CSV metadata folder into an sqlite database and .xlsx file
"""
import sqlite3
import pandas as pd
import argparse
from pathlib import Path

def main(args):
    # Connect to SQLite database
    data_folder = Path(args.data_path)
    if not data_folder.exists():
        raise ValueError("'data_path' must be a valid folder on disk")

    excel_writer = pd.ExcelWriter(args.outfile_excel, engine='xlsxwriter')
    conn = sqlite3.connect(args.outfile_sqlite)
    for csv_file in Path(args.data_path).glob("*.csv"):
        table_name = csv_file.stem
        df = pd.read_csv(csv_file.open())

        # Write the data to a sqlite table and excel sheet
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        df.to_excel(excel_writer, sheet_name=table_name, index=False)

    # Close sqlite connection
    conn.close()

    # Save and close Excel writer
    excel_writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=".")
    parser.add_argument("--outfile_sqlite", default="RiksdagenPersons.db")
    parser.add_argument("--outfile_excel", default="RiksdagenPersons.xlsx")
    main(parser.parse_args())
