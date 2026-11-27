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

def scrape_motion(path, url_root):
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
    body = root.findall(f".//{TEI_NS}body")[0]
    title_seen = False

    if len(sbs) == 1:
        for elem in sbs[0].iter():
            if elem.get("type") == "signature":
                if elem.get("who", "unknown") != "unknown":
                    metadata["signatures"] = metadata.get("signatures", []) + [elem.get("who", "unknown")]

    elif len(sbs) >= 2:
        
        for elem in body.iter():
            if not title_seen and elem.get("type") == "titleString":
                title_seen = True

            if title_seen and elem.tag.split("}")[-1] == "signatureBlock":
                for subelem in elem.iter():
                    if subelem.get("who", "unknown") != "unknown":
                        metadata["signatures"] = metadata.get("signatures", []) + [subelem.get("who", "unknown")]

    if len(sbs) >= 2 and not title_seen:
        LOGGER.warning("Correct signature block not found (multiple signatureBlocks and no title found)")
    text = [" ".join(t.split()) for t in body.itertext()]
    metadata["text"] = "\n\n".join(text)

    # Get link to the motion
    pb = root.find(f".//{TEI_NS}pb")
    if pb is not None:
        facs = pb.get("facs")
        old_facs = facs
        facs = "/".join(facs.split("/")[:-1])
        facs = facs.replace("riksdagen-motions-pdf/data/", "")
        facs = facs.replace("riksdagen-motions-pdf/", "")
        if "reg" not in facs:
            delimiter = "__"
            if delimiter not in facs:
                delimiter = "_"
            s = facs.split(delimiter)
            facs = delimiter.join(s[:-1]) + delimiter + s[-1].zfill(5) + "/"
        else:
            LOGGER.warning(f"Weird facs: {facs}")
        LOGGER.debug(f"{old_facs} -> {facs}")
        metadata["url"] = url_root + "/" + facs
    else:
        

    return metadata


def main(args):
    data = []
    for path in tqdm(args.motions):
        motion_data = scrape_motion(path, args.url_root)
        data.append(motion_data)

    df = pl.DataFrame(data)
    df = df.select(['motion', 'year', 'number', 'chamber', 'title', 'signatures', 'text', 'url'])
    df = df.sort("year", "chamber", "number", "motion")

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
        df = df.select(['motion', 'year', 'number', 'chamber', 'title', 'text', 'url'])
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
    parser.add_argument("--url_root", type=str, default="https://pdf.swedeb.se/riksdagen-motions-pdf")
    args = parser.parse_args()
    LOGGER.info(f"Args: {args}")
    args = impute_args(args)
    main(args)