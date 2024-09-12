"""
Find margin notes with a date in protocols and include them as metadata.
"""
from lxml import etree
from pyriksdagen.refine import detect_date
from pyriksdagen.utils import (
    infer_metadata,
    parse_protocol,
    protocol_iterators,
    write_protocol,
)
import progressbar
import argparse




def main(args):

    if args.protocol:
        protocols = [args.protocol]
    else:
        if args.records_folder is not None:
            data_location = args.records_folder
        else:
            data_location = get_data_location("records")
        protocols = sorted(list(protocol_iterators(data_location,
                                                    start=args.start,
                                                    end=args.end)))

    for protocol_path in progressbar.progressbar(protocols):
        metadata = infer_metadata(protocol_path)
        root = parse_protocol(protocol_path)
        root, dates = detect_date(root, metadata)

        write_protocol(root, protocol_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--start", type=int, default=1920, help="Start year")
    parser.add_argument("-e", "--end", type=int, default=2022, help="End year")
    parser.add_argument("-r", "--records-folder",
                        type=str,
                        default=None,
                        help="(optional) Path to records folder, defaults to environment var or `data/`")
    parser.add_argument("-p", "--protocol",
                        type=str,
                        default=None,
                        help="operate on a single protocol")
    args = parser.parse_args()
    main(args)
