"""
Add a randomly generated UUID to all elements in the XML ID field that are currently missing one.
"""
from lxml import etree
from pathlib import Path
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
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)



def add_protocol_id(protocol):
    ids = set()
    num_ids = 0

    root, ns = parse_protocol(protocol, get_ns=True)

    body = root.find(f".//{ns['tei_ns']}body")
    if body is None:
        print(protocol)
    else:
        divs = body.findall(f"{ns['tei_ns']}div")
        for div in divs:
            protocol_id = Path(protocol).stem
            seed_str = f"{protocol_id}\n{' '.join(div.itertext())}"
            x = div.attrib.get(f"{ns['xml_ns']}id", get_formatted_uuid(seed_str))
            div.attrib[f"{ns['xml_ns']}id"] = x
            num_ids += 1
            ids.add(x)

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

    protocols = args.records
    num_ids = 0
    ids = []
    with multiprocessing.Pool() as pool:
        for i, n in tqdm(pool.imap(add_protocol_id, protocols), total=len(protocols)):
            ids += i
            num_ids += n

        assert len(set(ids)) == num_ids




if __name__ == "__main__":
    parser = fetch_parser("records")
    args = impute_args(parser.parse_args())
    main(args)
