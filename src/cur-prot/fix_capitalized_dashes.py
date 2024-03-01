"""
Concatenate split names of format "PERS- SON" into "PERSSON"
"""
from pyparlaclarin.read import paragraph_iterator
from pyriksdagen.utils import (
    get_data_location,
    parse_protocol,
    protocol_iterators,
    write_protocol,
)
from tqdm import tqdm
import argparse, re




def main(args):
    # NB: [A-ZÀ-Þ] is UPPERCASE LETTERS + Accented UPPERCASE letters, ÅÄÖ etc
    pattern = "([A-ZÀ-Þ]{2,10})(- )([A-ZÀ-Þ]{2,10})"
    e = re.compile(pattern)

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

    for protocol in tqdm(protocols, total=len(protocols)):
        root = parse_protocol(protocol)
        for elem in paragraph_iterator(root, output="lxml"):
            txt = elem.text
            if txt is not None and len(e.findall(txt)) > 0:
                elem.text = re.sub(pattern, r"\1\3", txt)

        write_protocol(root, protocol)




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
