"""
Query wikidata for metadata, process it and save it in corpus/metadata
"""
from pathlib import Path
from pyriksdagen.data import queries as pyriksdagen_queries
from pyriksdagen.db import clean_person_duplicates
from pyriksdagen.utils import get_data_location
from pyriksdagen.wikidata import (
    query2df,
    separate_name_location,
    move_party_to_party_df,
    elongate_external_ids,
)
from SPARQLWrapper import SPARQLWrapper, JSON
import argparse
import numpy as np
import os
import pandas as pd
import polars as pl
import re, time
import warnings
from trainerlog import get_logger
LOGGER = get_logger("wikidata-query")

def track_missing_id(df, l, id_map=None):
    print(df)
    df = df.to_pandas()
    no_id = df.loc[pd.isna(df["person_id"])]
    for i, r in no_id.iterrows():
        tmpdf = df.loc[(df['government'] == r['government']) & (df['role'] == r['role']) & (df['wiki_id'] == r['wiki_id']) & (pd.notnull(df["person_id"]))]
        if tmpdf.empty:
            found = False
            if not id_map.empty:
                tmpidmap = id_map.loc[id_map['wiki_id'] == r['wiki_id']].copy()
                if not tmpidmap.empty:
                    tmpidmap.reset_index(drop=True, inplace=True)
                    swerik_id = tmpidmap.at[0, "person_id"]
                    df.at[i, "person_id"] = swerik_id
                    found = True
            if found == False:
                if r['wiki_id'] not in l:
                    l.append(r['wiki_id'])

    df = pl.from_pandas(df)
    df = df.filter(pl.col('person_id').is_not_null())
    df = df.drop("wiki_id")
    return df, l


def postprocess_riksmote(df, metadata_folder):
    # Only EK is scraped, hard-code the value
    df = df.with_columns(pl.lit("ek").alias("chamber"))

    # Take everything up to 1990 from the manually annotated file
    riksdagen_year = pl.read_csv(f'{metadata_folder}/riksdag-year.csv')
    riksdagen_year = riksdagen_year.filter(pl.col('start') < "1990-12-31")

    # Drop everything before 1990 from the scraped df
    df = df.filter(pl.col('start') >= "1990-12-31")
    df = df.with_columns(pl.col("parliament_year").cast(pl.Int64))
    # Make sure the columns match
    df = df.select(riksdagen_year.columns)
    joint_df = pl.concat([riksdagen_year, df])
    return joint_df

def main(args):
    if args.metadata_folder is None:
        metadata_folder = get_data_location("metadata")
    else:
        metadata_folder = args.metadata_folder

    if args.query_folder is None:
        query_folder = pyriksdagen_queries.__path__._path[0]
    else:
        query_folder = args.query_folder

    if args.queries is None:
        queries = sorted([q.stem for q in Path(query_folder).glob('*.rq')])
    else:
        queries = args.queries
    input_folders = ['name_location_specifier', 'alias', "member_of_parliament", "party_affiliation"]

    # Query for and store cleaned versions of metadata
    d = {}
    no_swerik_id = []
    id_map = None
    if  "wiki_id" in queries:
        LOGGER.info(f"Query Wiki ID started.")
        id_map = query2df("wiki_id", args.source)
        LOGGER.debug(f"id map type: {type(id_map)}")
        id_map = id_map.unique()
        id_map.write_csv(f'{metadata_folder}/wiki_id.csv')


    for q in queries:
        LOGGER.info(f"Query: {q}")
        if q == "wiki_id":
            continue
        LOGGER.info(f"Query {q} started.")
        df = query2df(q, args.source)
        LOGGER.info(f"Obtained df:\n{df}")
        # Format values
        if 'riksdagen_id' in df.columns:
            df = df.with_columns(pl.col('riksdagen_id').cast(pl.String))

        if 'gender' in df.columns:
            df = df.with_columns(pl.col("gender").replace({'kvinna':'woman', 'man':'man'}))

        if q == 'minister':
            df = df.with_columns(pl.col("role").str.replace('Sveriges', '', literal=True).str.strip_chars())
            df, no_swerik_id = track_missing_id(df, no_swerik_id, id_map=id_map)

        if q == 'member_of_parliament':
            df = df.with_columns(pl.col("role").str.extract(r'([A-Za-zÀ-ÿ]*ledamot)'))
            df, no_swerik_id = track_missing_id(df, no_swerik_id, id_map=id_map)

        if q == 'speaker':
            df, no_swerik_id = track_missing_id(df, no_swerik_id, id_map=id_map)

        if q == "external_identifiers":
            df = elongate_external_ids(df)

        if q == "government":
            sveriges_regering = pl.col("government") == "Sveriges regering"
            hardcoded_object_len = len(df.filter(sveriges_regering))
            if hardcoded_object_len == 0:
                LOGGER.warning("Hard coded value 'Sveriges regering' not found in the query results")

            df = df.filter(~sveriges_regering)
            LOGGER.info(f"Filtered df:\n{df}")

        if 'riksmote' in df.columns:
            # Remove redundant information
            LOGGER.info(f"Before:\n{df}")
            df = df.with_columns(pl.col("riksmote").str.replace("Riksmötet ", ""))
            df = df.with_columns(pl.col("riksmote").str.replace("riksmötet ", ""))
            df = df.with_columns(pl.col("riksmote").str.replace("/20", "", literal=True))
            df = df.with_columns(pl.col("riksmote").str.replace("/19", "", literal=True))
            LOGGER.info(f"After:\n{df}")

            # Separate meeting, e.g. "202223" from the optional specifier eg. "urtima"
            df = df.with_columns(pl.col("riksmote").str.extract(r"(\d+)", 1).alias("parliament_year"))
            df = df.with_columns(pl.col("riksmote").str.extract(r"(\D+)", 1).str.replace(" ", "").alias("specifier"))
            LOGGER.info(f"After:\n{df}")

            # Convert column name to conform to standard
            df = df.rename({"riksmote_id": "wiki_id"})

            # Drop unnecessary columns
            df = df.select("parliament_year", "specifier", "start", "end", "wiki_id")

            df = postprocess_riksmote(df, metadata_folder)

        # Store files needing additional preprocessing in input
        folder = metadata_folder if not q in input_folders else args.input_metadata_folder
        if folder == args.input_metadata_folder:
            d[q] = df

        if q == 'person':
            df = clean_person_duplicates(df)

        df = df.unique()
        df.write_csv(f'{folder}/{q}.csv')
        time.sleep(0.2)

    # Process name and location files
    if d:
        for key in d.keys():
            if key not in queries:
                d['key'] = pd.read_csv(f'{args.input_metadata_folder}/{key}.csv')
        name, loc = separate_name_location(d['name_location_specifier'], d['alias'])
        name.write_csv(f'{metadata_folder}/name.csv')
        loc.write_csv(f'{metadata_folder}/location_specifier.csv')

        mp_df, party_df = move_party_to_party_df(d['member_of_parliament'], d['party_affiliation'])
        mp_df.write_csv(f'{metadata_folder}/member_of_parliament.csv')
        party_df.write_csv(f'{metadata_folder}/party_affiliation.csv')

    if len(no_swerik_id) > 0:
        LOGGER.warning("Some entities returned in the queries seem not to have a swerik ID. Check and add an ID, then requery.")
    with open(f"{args.input_metadata_folder}/no_swerik_id_query_results.txt", "w+") as outf:
        [outf.write(f"{_}\n") for _ in no_swerik_id]




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_metadata_folder', type=str, default="input/metadata")
    parser.add_argument('--metadata_folder', type=str, default=None)
    parser.add_argument('--query_folder', type=str, default=None)
    parser.add_argument('-q', '--queries',
                        default=None,
                        nargs='+',
                        help='One or more sparql query files (separated by space)')
    parser.add_argument('-s', '--source',
                        default=None,
                        nargs='+',
                        help='One or more of member_of_parliament | minister | speaker (separated by space)')
    args = parser.parse_args()
    main(args)
