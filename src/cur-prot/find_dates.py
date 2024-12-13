"""
Find margin notes with a date in protocols and include them as metadata.
"""
from lxml import etree
from pyriksdagen.refine import detect_date
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from pyriksdagen.utils import (
    infer_metadata,
    parse_tei,
    write_tei,
)
from tqdm import tqdm
import argparse




def main(args):

    for record in tqdm(args.records):
        metadata = infer_metadata(record)
        root, ns = parse_tei(record)
        root, dates = detect_date(root, metadata, skip_doctors_notes=args.skip_doctors_notes)
        write_tei(root, record)




if __name__ == "__main__":
    parser = fetch_parser("records")
    parser.add_argument("--skip-doctors-notes", action='store_true')
    main(impute_args(parser.parse_args()))
