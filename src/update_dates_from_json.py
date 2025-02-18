"""
Replace document dates with Riksdagens Öppna data JSON metadata
"""
from lxml import etree
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from pyriksdagen.utils import (
    infer_metadata,
    parse_tei,
    write_tei,
    TEI_NS
)
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import warnings

def read_in_json(paths):
    json_paths = []
    print(f"Read in {len(paths)} folders of JSON files...")
    for path in paths:
        json_folder = Path(path)
        json_paths = json_paths + list(json_folder.glob("*.json"))

    ids_to_dates = {}
    for file in tqdm(sorted(json_paths)):
        with file.open(encoding='utf-8-sig') as f:
            d = json.load(f)
        metadata = d["dokumentstatus"]["dokument"]
        date = metadata["datum"]
        meeting = metadata["rm"]
        number = int(metadata["beteckning"])
        # 1997-12-12
        date = date.strip().split()[0]
        meeting = meeting.replace("/", "")
        # prot-199798--113.xml
        protocol_id = f"prot-{meeting}--{number:03d}"
        ids_to_dates[protocol_id] = date

    return ids_to_dates

def update_date(root, new_date):
    for text in root.findall(".//" + TEI_NS + "text"):
        for front in text.findall(".//" + TEI_NS + "front"):
            # Remove old docDates
            for docDate in front.findall(".//" + TEI_NS + "docDate"):
                docDate.getparent().remove(docDate)
            for div in front.findall(".//" + TEI_NS + "div"):
                for docDate in div.findall(".//" + TEI_NS + "docDate"):
                    docDate.getparent().remove(docDate)

            # Add new docDates
            for div in front.findall(".//" + TEI_NS + "div"):
                if div.attrib.get("type") == "preface":
                    for docDate in div.findall(".//" + TEI_NS + "docDate"):
                        docDate.getparent().remove(docDate)

                    docDate = etree.SubElement(div, "docDate")
                    docDate.text = new_date
                    docDate.attrib["when"] = new_date
    return root

def main(args):
    ids_to_dates = read_in_json(args.json_path)
    for record in tqdm(args.records):
        metadata = infer_metadata(record)
        protocol_id = metadata["protocol"].replace("_", "-")
        new_date = ids_to_dates.get(protocol_id)
        if new_date is not None:
            root, ns = parse_tei(record)
            root = update_date(root, new_date)
            write_tei(root, record)
        else:
            warnings.warn(f"No date found for {protocol_id}")

if __name__ == "__main__":
    parser = fetch_parser("records")
    parser.add_argument("--json_path", type=str, default=[], nargs="+")
    parser.add_argument("--skip-doctors-notes", action='store_true')
    main(impute_args(parser.parse_args()))
