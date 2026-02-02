"""
Find introductions in the protocols. After finding an intro,
tag the next paragraph as an utterance.
"""
import pandas as pd
import progressbar
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from pyriksdagen.db import load_patterns
from pyriksdagen.io import (
    parse_tei,
    write_tei
)
from pyriksdagen.refine import (
    find_introductions,
)
from pyriksdagen.utils import (
    infer_metadata,
)

def main(args):
    protocols = args.records
    intro_df = pd.read_csv(args.segmentation_file)

    for protocol in progressbar.progressbar(protocols):
        intro_ids = intro_df.loc[intro_df['file_path'] == protocol, 'id'].tolist()

        metadata = infer_metadata(protocol)
        protocol_id = protocol.split("/")[-1]
        year = metadata["year"]

        root, ns = parse_tei(protocol)

        years = [
            int(elem.attrib.get("when").split("-")[0])
            for elem in root.findall(
                ".//{http://www.tei-c.org/ns/1.0}docDate"
            )
        ]

        if not year in years:
            year = years[0]
        
        pattern_db = load_patterns()
        pattern_db = pattern_db[
            (pattern_db["start"] <= year) & (pattern_db["end"] >= year)
        ]
        root = find_introductions(root, pattern_db, intro_ids, minister_db=None, remove_missing=args.remove_negative)

        write_tei(root, protocol)




if __name__ == "__main__":
    parser = fetch_parser("records")
    parser.add_argument("--segmentation_file",
                        type=str,
                        default="input/segmentation/intros.csv")
    parser.add_argument("--remove_negative", action="store_true",
                        help="Also remove previously detected 'speaker' attributes. By default only finds new intros.")
    args = impute_args(parser.parse_args())
    main(args)
