"""
Export the motions to newline delimited JSON and/or sqlite

The NDJSON output is one table while the sqlite has two tables: motions and signatures
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

def scrape_motion(path):
    root, _ = parse_tei(path, get_ns=True)
    # Get protocol metadata
    record_id = root.attrib[f"{XML_NS}id"]
    metadata = infer_metadata(record_id)
    metadata["motion"] = record_id

    for teiHeader in root.findall(f".//{TEI_NS}teiHeader"):
        for title in teiHeader.findall(f".//{TEI_NS}title"):
            metadata["title"] = title.text


    # Get signatures
    sbs = list(root.findall(f".//{TEI_NS}signatureBlock"))
    metadata["has_signatures"] = len(sbs) >= 1

    # Due to duplication, only take from the first signatureBlock
    for sb in sbs[:1]:
        for elem in sb.iter():
            if elem.get("type") == "signature":
                if elem.get("who", "unknown") != "unknown":
                    metadata["signatures"] = metadata.get("signatures", []) + [elem.get("who", "unknown")]

    body = root.findall(f".//{TEI_NS}body")[0]
    text = [" ".join(t.split()) for t in body.itertext()]
    metadata["text"] = "\n\n".join(text)

    return metadata


def main(args):
    data = []
    motions = args.motions
    import random
    random.shuffle(motions)
    for path in tqdm(motions[:1000]):
        motion_data = scrape_motion(path)
        data.append(motion_data)

    df = pl.DataFrame(data)
    df = df.select(['motion', 'sitting', 'number', 'chamber', 'title', 'signatures', 'text'])
    df = df.sort("sitting", "chamber", "number", "motion")

    if "ndjson" in args.formats:
        LOGGER.info("Export to one ndjson file")
        df.write_ndjson(f"motions.ndjson")

    signature_df = df.select("motion", "signatures")
    signature_df = signature_df.filter(pl.col("signatures").is_not_null())
    signature_df = signature_df.explode("signatures")
    signature_df = signature_df.rename({"signatures": "person_id"})

    if "sqlite" in args.formats:
        LOGGER.info("Export to sqlite")
        if Path("motions.sqlite").exists():
            LOGGER.warning("Remove existing motions.sqlite...")
            Path("motions.sqlite").unlink()
        df = df.select(['motion', 'sitting', 'number', 'chamber', 'title', 'text'])
        df.write_database(
            table_name="motions",
            connection="sqlite:///motions.sqlite",
        )
        signature_df.write_database(
            table_name="motion_signatures",
            connection="sqlite:///motions.sqlite",
        )

if __name__ == "__main__":
    parser = fetch_parser("motions")
    parser.add_argument("--formats", type=str, nargs="+", default=["sqlite", "ndjson"])
    args = parser.parse_args()
    LOGGER.info(f"Args: {args}")
    args = impute_args(args)
    main(args)