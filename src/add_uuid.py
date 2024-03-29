"""
Add a randomly generated UUID to all elements in the XML ID attribute that are currently missing one.

Also adds the document ID (eg. prot-year--number) in the TEI element as an XML ID attribute if its missing.
"""
from lxml import etree
import argparse
from pyriksdagen.utils import elem_iter, protocol_iterators, get_formatted_uuid
from pyriksdagen.utils import TEI_NS, XML_NS
from tqdm import tqdm
import multiprocessing

def add_protocol_id(protocol):
    ids = set()
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.parse(protocol, parser).getroot()
    
    # Accomodate both TEI and teiCorpus root
    tei = root.find(f"{TEI_NS}TEI")
    if root.tag.split("}")[-1] == "TEI":
        tei = root

    # Set ID for TEI element to be the filename
    if f"{XML_NS}id" not in tei.attrib:
        tei.attrib[f"{XML_NS}id"] = protocol.split("/")[-1][:-4]

    # Create UUIDs for other elements
    num_ids = 0
    for tag, elem in elem_iter(root):
        if tag == "u":
            for subelem in elem:
                x = subelem.attrib.get(f'{XML_NS}id', get_formatted_uuid())
                subelem.attrib[f'{XML_NS}id'] = x
                ids.add(x)
                num_ids += 1
            x = elem.attrib.get(f'{XML_NS}id', get_formatted_uuid())
            elem.attrib[f'{XML_NS}id'] = x
            ids.add(x)
            num_ids += 1
                
        elif tag in ["note"]:
            x = elem.attrib.get(f'{XML_NS}id', get_formatted_uuid())
            elem.attrib[f'{XML_NS}id'] = x
            ids.add(x)
            num_ids += 1

    for body in root.findall(f".//{TEI_NS}body"):
        for div in body:
            elem_id_list = [elem.attrib.get(f'{XML_NS}id') for elem in div]
            elem_id_list = [elem_id for elem_id in elem_id_list if elem_id is not None]
            elem_id_list = '\n'.join(elem_id_list)
            seed_str =  f"div\n{elem_id_list}"
            new_div_id = get_formatted_uuid(seed_str)
            if f'{XML_NS}id' not in div.attrib:
                print(seed_str, new_div_id)
            x = div.get(f'{XML_NS}id', new_div_id)
            div.attrib[f'{XML_NS}id'] = x
            ids.add(x)
            num_ids += 1

    b = etree.tostring(
        root, pretty_print=True, encoding="utf-8", xml_declaration=True
    )
    f = open(protocol, "wb")
    f.write(b)

    assert len(ids) == num_ids
    return ids, num_ids


def main(args):
    protocols = sorted(list(protocol_iterators(args.records_folder, start=args.start, end=args.end)))
    num_ids = 0
    ids = []
    with multiprocessing.Pool() as pool:
        for i, n in tqdm(pool.imap(add_protocol_id, protocols), total=len(protocols)):
            ids += i
            num_ids += n

        assert len(set(ids)) == num_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records_folder", type=str, default="corpus/records")
    parser.add_argument("-s", "--start", type=int, default=None, help="Start year")
    parser.add_argument("-e", "--end", type=int, default=None, help="End year")
    args = parser.parse_args()
    main(args)
