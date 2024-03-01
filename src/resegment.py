"""
Find introductions in the protocols. After finding an intro,
tag the next paragraph as an utterance.
"""
from pyparlaclarin.refine import format_texts
from pyriksdagen.db import load_patterns
from pyriksdagen.refine import (
    detect_mps,
    find_introductions,
    update_ids,
)
from pyriksdagen.utils import (
    infer_metadata,
    get_data_location,
    protocol_iterators,
    write_protocol,
)
from lxml import etree
import pandas as pd
import os, progressbar, argparse




def main(args):
    if args.protocol:
        protocols = [args.protocol]
    else:
        start_year = args.start
        end_year = args.end
        protocols = list(protocol_iterators(
                            get_data_location("records"),
                            start=args.start,
                            end=args.end))
    parser = etree.XMLParser(remove_blank_text=True)
    intro_df = pd.read_csv(args.segmentation_file)

    for protocol in progressbar.progressbar(protocols):
        intro_ids = intro_df.loc[intro_df['file_path'] == protocol, 'id'].tolist()

        metadata = infer_metadata(protocol)
        protocol_id = protocol.split("/")[-1]
        year = metadata["year"]
        root = etree.parse(protocol, parser).getroot()

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
        root = find_introductions(root, pattern_db, intro_ids, minister_db=None)

        write_protocol(root, protocol)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation_file", type=str, default="input/segmentation/intros.csv")
    parser.add_argument("-s", "--start", type=int, default=1920, help="Start year")
    parser.add_argument("-e", "--end", type=int, default=2022, help="End year")
    parser.add_argument("-p", "--protocol",
                        type=str,
                        default=None,
                        help="operate on a single protocol")
    args = parser.parse_args()
    main(args)
