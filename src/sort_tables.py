"""
Sort CSV files according to sort-order.json
"""
import polars as pl
import argparse
import tqdm
import json
from pathlib import Path
from trainerlog import get_logger
LOGGER = get_logger("sort-order")

def fetch_sort_order(filepath):
    with open(filepath) as f: 
        sort_order =  json.load(f)
    return sort_order

def main(args):
    sort_order = fetch_sort_order(args.sort_order_path)
    data_folder = Path(args.data_path)

    for fpath in tqdm.tqdm(list(data_folder.glob("*.csv"))):
        fname = fpath.stem + ".csv"
        val = sort_order[fname]
        sort_keys = None
        descending = False
        if isinstance(val, list):
            sort_keys = val
        else:
            sort_keys = val["columns"]

        df = pl.read_csv(fpath, infer_schema_length=10000)
        df = df.sort(sort_keys)
        df.write_csv(fpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--sort_order_path", type=str, default="test/data/sort-order.json")
    args = parser.parse_args()
    main(args)
