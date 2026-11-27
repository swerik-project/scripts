#!/usr/bin/env python3
"""
Scrape full names of politicians from Wikipedia introductions
"""
from lxml import etree
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from pyriksdagen.utils import (
    get_data_location,
)
from tqdm import tqdm
import polars as pl
import re
from trainerlog import get_logger
from pathlib import Path
import requests
import time
import numpy as np

LOGGER = get_logger("map-signatures")
HEADERS={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}

def get_wikipedia_url(person_id, wiki_id):
    url = f"https://www.wikidata.org/wiki/{wiki_id}"
    response = requests.get(url, headers=HEADERS)
    LOGGER.info(f"Return code: {response.status_code}")
    LOGGER.debug(f"Content:\n{response.content}")

    content = response.content.decode("utf-8")
    if person_id not in content:
        LOGGER.error(f"person_id: {person_id} not found at {url}")
        return None
    else:
        LOGGER.info(f"person_id: {person_id} found at {url}")
    parser = etree.HTMLParser()
    html_root = etree.fromstring(response.content, parser)
    wikipedia_url = None
    for a in html_root.findall(".//a"):
        if "sv.wikipedia" in a.get("href", ""):
            wikipedia_url = a.get("href", "")

    return wikipedia_url

def get_name_from_wikitext(wikipedia_url):
    params = {
    'action': "edit",
    }
    response = requests.get(wikipedia_url, headers=HEADERS, params=params)
    LOGGER.info(f"Return code: {response.status_code}")
    LOGGER.debug(f"Content:\n{response.content}")

    content = response.content.decode("utf-8")
    parser = etree.HTMLParser()
    html_root = etree.fromstring(response.content, parser)

    for div in html_root.findall(".//div"):
        if div.get("id", "") == "mw-content-text":
            
            for t in " ".join(div.itertext()).split("\n"):
                if "'''" in t:
                    return t

NAME_REGEX = re.compile("'''.{10,55}'''")
def parse_name(name_str):
    m = NAME_REGEX.search(name_str)
    if m is not None:
        return m.group(0).replace("'", "")

def main(args):
    metadata_location = get_data_location("metadata")

    output_file = Path(args.output_file)
    current_names_data = ""
    if output_file.exists():
        LOGGER.info(f"Read in previously downloaded data from {output_file}...")
        current_names_data = output_file.open().read()
    else:
        LOGGER.info(f"No previously downloaded data in {output_file}, skip.")

    db = pl.read_csv(Path(metadata_location)/ "name.csv")
    wiki_db = pl.read_csv(Path(metadata_location)/ "wiki_id.csv")
    db = db.join(wiki_db, on="person_id", how="left")
    db = db.with_columns(pl.col("name").str.split(" ").list.len().alias("no_names"))
    db = db.sort("person_id", "no_names", descending=True).unique("person_id", keep="first")
    db = db.filter(pl.col("no_names") <= 2)
    db = db.filter(pl.col("wiki_id").is_not_null())
    LOGGER.info(f"Looping over db:\n{db}")



    for row in tqdm(db.to_dicts()):
        person_id = row["person_id"]
        if not person_id in current_names_data:
            wikipedia_url = get_wikipedia_url(row["person_id"], row["wiki_id"])
            LOGGER.info(f"Wikipedia URL: {wikipedia_url}")

            waittime = np.exp(np.random.randn() * 1 + 1.5)
            LOGGER.info(f"Wait for {waittime} seconds")
            time.sleep(waittime)

            if wikipedia_url is not None:
                name_str = get_name_from_wikitext(wikipedia_url)
                LOGGER.info(f"Full name str {name_str}")
                name = parse_name(name_str)
                LOGGER.info(f"Name {name}")

                waittime = np.exp(np.random.randn() * 1 + 1.5)
                LOGGER.info(f"Wait for {waittime} seconds")
                time.sleep(waittime)

                if name is not None:
                    text_row = row["person_id"] + "," + name + "\n"

                    with output_file.open("a") as myfile:
                        myfile.write(text_row)
        else:
            LOGGER.warning(f"Skip {person_id}, already exists in {output_file}")





if __name__ == '__main__':
    parser = fetch_parser("motions", docstring=__doc__)
    parser.add_argument("--metadata-location",
                        type=str,
                        default="input/metadata/db.parquet",
                        help="path to compiled metadata db")
    parser.add_argument("--recompile-metadata", type=bool, default=True)
    parser.add_argument("--write-compiled-db",
                        type=str,
                        default="True",
                        choices=["True", "False"],
                        help="write db after compile")
    parser.add_argument("--output-file",
                        type=str,
                        default="wikipedia_names.txt"
                        )

    args = parser.parse_args()
    LOGGER.info(f"args: {args}")
    main(impute_args(args))
