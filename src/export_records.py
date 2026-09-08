"""
Export the speeches in the records to newline delimited JSON and/or sqlite

The NDJSON output is flattened in the sense that each row has record level metadata
while the sqlite is not and has two tables
"""
from pyriksdagen.utils import (
    infer_metadata
)
from pyriksdagen.utils import (
    TEI_NS,
    XML_NS
)
from pyriksdagen.io import (
    parse_tei,
)
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from trainerlog import get_logger
from tqdm import tqdm
import polars as pl
from pathlib import Path

LOGGER = get_logger(name="export-records")

def scrape_record(record):
    root, _ = parse_tei(record, get_ns=True)
    # Get protocol metadata
    record_id = root.attrib[f"{XML_NS}id"]
    metadata = infer_metadata(record_id)
    metadata["record"] = record_id

    for front in root.findall(f".//{TEI_NS}front"):
        for docDate in front.findall(f".//{TEI_NS}docDate"):
            date = docDate.attrib["when"]
            if metadata.get("start_date") is None:
                metadata["start_date"] = date
                metadata["end_date"] = date

            if metadata.get("start_date") > date:
                metadata["start_date"] = date
            if metadata.get("end_date") < date:
                metadata["end_date"] = date


    # Get speeches
    speeches = {}
    all_u_ids = set()
    for textDesc in root.findall(f".//{TEI_NS}textDesc"):
        for constitution in textDesc.findall(f".//{TEI_NS}constitution"):
            speech_index = 0
            for speech_note in constitution:
                speech_id = speech_note.attrib[f"{XML_NS}id"]
                #print(speech_id)

                # scrape u tags from linkGrp
                u_ids = set()
                for ptr in speech_note.findall(f".//{TEI_NS}ptr"):
                    u_id = ptr.attrib["target"].replace("#", "")
                    u_ids.add(u_id)
                    all_u_ids.add(u_id)

                speeches[speech_id] = {"record": record_id, "u_ids": u_ids, "who": None, "text": None, "ix": speech_index}
                speech_index += 1

    if len(speeches) == 0:
        return None, metadata

    for u in root.findall(f".//{TEI_NS}u"):
        u_id = u.attrib[f"{XML_NS}id"]
        if u_id in all_u_ids:
            LOGGER.debug(f"u {u_id} in all u ids")
            speech = None
            for i in speeches:
                if u_id in speeches[i]["u_ids"]:
                    speech = i

            LOGGER.debug(f"u {u_id} belongs to speech: {speech}")
            who = u.attrib["who"]
            if who == "unknown":
                who = None
            speeches[speech]["who"] = who
            for seg in u:
                text = " ".join(seg.text.split())
                if speeches[speech]["text"] is None:
                    speeches[speech]["text"] = text
                else:
                    speeches[speech]["text"] += "\n\n" + text



    speech_list = []
    for speech_id in speeches:
        speech_dict = speeches[speech_id]
        speech_dict["speech"] = speech_id
        speech_list.append(speech_dict)

    df = pl.DataFrame(speech_list, infer_schema_length=None)
    df = df.select("speech", "record", "ix", "who", "text")

    # Make sure who is pl.String in case all who's happen to be null
    df = df.with_columns(pl.col("who").cast(pl.String))
    return df, metadata


def main(args):
    all_dfs = []
    record_metadata = []
    for record in tqdm(args.records):
        df, metadata = scrape_record(record)
        record_metadata.append(metadata)
        if df is None:
            LOGGER.warning(f"No speeches in {record}")
        else:
            all_dfs.append(df)

    df = pl.concat(all_dfs)
    df = df.sort("record", "ix")
    df = df.rename({"ix": "speech_number"})
    df = df.select("speech", "record", "who", "text", "speech_number")
    df = df.with_columns(pl.col("speech_number") + 1)

    metadata_df = pl.DataFrame(record_metadata)
    metadata_df = metadata_df.rename({"sitting": "session"})
    metadata_df = metadata_df.select("record", "session", "chamber", "number", "start_date", "end_date")
    metadata_df = metadata_df.sort("session", "chamber", "number")

    if "sqlite" in args.formats:
        LOGGER.info("Export to sqlite")
        if Path("records.sqlite").exists():
            Path("records.sqlite").unlink()
        df.write_database(
            table_name="speeches",
            connection="sqlite:///records.sqlite",
        )
        metadata_df.write_database(
            table_name="records",
            connection="sqlite:///records.sqlite",
        )

    # Flattened formats
    df = df.join(metadata_df, on="record")
    df = df.sort("session", "chamber", "number", "speech_number")
    df = df.with_columns(pl.col("session").str.head(3).alias("decade"))
    df = df.rename({"number": "record_number"})

    if "ndjson-decade" in args.formats:
        LOGGER.info("Export to ndjson by decade")
        for decade in sorted(set(df["decade"])):
            df_decade = df.filter(pl.col("decade") == decade)
            df_decade_columns = [col for col in df_decade.columns if col != "decade"]
            df_decade = df_decade.select(df_decade_columns)
            LOGGER.info(f"{decade}:\ndf_decade")
            df_decade.write_ndjson(f"records_speeches_{decade}0s.ndjson")
    if "ndjson" in args.formats:
        LOGGER.info("Export to one ndjson file")
        df_decade_columns = [col for col in df.columns if col != "decade"]
        df_decade = df.select(df_decade_columns)
        df_decade.write_ndjson(f"records_speeches.ndjson")

if __name__ == "__main__":
    parser = fetch_parser("records")
    parser.add_argument("--formats", type=str, nargs="+", default=["sqlite", "ndjson"])
    args = parser.parse_args()
    LOGGER.info(f"Args: {args}")
    args = impute_args(args)
    main(args)