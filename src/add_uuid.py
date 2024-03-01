"""
Add a randomly generated UUID to all elements in the XML ID field that are currently missing one.
"""
from lxml import etree
from pyriksdagen.utils import (
    elem_iter,
    get_formatted_uuid,
    get_data_location,
    parse_protocol,
    protocol_iterators,
    write_protocol,
)
from tqdm import tqdm
import argparse
import multiprocessing




def add_protocol_id(protocol):

    ids = set()
    num_ids = 0

    root, ns = parse_protocol(protocol, get_ns=True)
    for tag, elem in elem_iter(root):
        if tag == "u":
            for subelem in elem:
                x = subelem.attrib.get(f"{ns['xml_ns']}id", get_formatted_uuid())
                subelem.attrib[f"{ns['xml_ns']}id"] = x
                ids.add(x)
                num_ids += 1
            x = elem.attrib.get(f"{ns['xml_ns']}id", get_formatted_uuid())
            elem.attrib[f"{ns['xml_ns']}id"] = x
            ids.add(x)
            num_ids += 1
        elif tag in ["note"]:
            x = elem.attrib.get(f"{ns['xml_ns']}id", get_formatted_uuid())
            elem.attrib[f"{ns['xml_ns']}id"] = x
            ids.add(x)
            num_ids += 1

    write_protocol(root, protocol)

    assert len(ids) == num_ids
    return ids, num_ids




def main(args):

    num_ids = 0
    ids = []

    if args.protocol:
        protocols = [args.protocol]
    else:
        protocols = sorted(list(protocol_iterators(
                                        get_data_location("records"),
                                        start=args.start, end=args.end)))

    with multiprocessing.Pool() as pool:
        for i, n in tqdm(pool.imap(add_protocol_id, protocols), total=len(protocols)):
            ids += i
            num_ids += n

        assert len(set(ids)) == num_ids




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records_folder", type=str, default="corpus/records")
    parser.add_argument("-s", "--start", type=int, default=1920, help="Start year")
    parser.add_argument("-e", "--end", type=int, default=2022, help="End year")
    parser.add_argument("-p", "--protocol",
                        type=str,
                        default=None,
                        help="operate on a single protocol")
    args = parser.parse_args()
    main(args)
