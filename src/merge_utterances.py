"""
Merge consecutive 'u' elements
"""
from lxml import etree
import argparse
from pyriksdagen.utils import corpus_iterator
from tqdm import tqdm
from pyriksdagen.utils import TEI_NS
from pyriksdagen.io import (
    parse_tei,
    write_tei
)

def merge_us(root):
    for body in root.findall(f".//{TEI_NS}body"):
        for div in body.findall(f".//{TEI_NS}div"):
            previous_u = None
            for elem in div:
                if elem.tag.split("}")[-1] == "u":
                    if previous_u is None:
                        previous_u = elem
                    else:
                        for seg in elem:
                            previous_u.append(seg)
                        elem.getparent().remove(elem)
                else:
                    previous_u = None
    return root

def remove_dead_nextprev_links(root):
    all_ids = set(root.xpath('//@xml:id', namespaces={"xml": "http://www.w3.org/XML/1998/namespace"}))
    for body in root.findall(f".//{TEI_NS}body"):
        for div in body.findall(f".//{TEI_NS}div"):
            for elem in div:
                if elem.tag.split("}")[-1] == "u":
                    if elem.attrib.get("next") not in all_ids and "next" in elem.attrib:
                        del elem.attrib["next"]
                    if elem.attrib.get("prev") not in all_ids and "prev" in elem.attrib:
                        del elem.attrib["prev"]
    return root

def main(args):
    protocols = sorted(list(corpus_iterator("prot", args.records_folder, start=args.start, end=args.end)))
    for p in tqdm(protocols):
        root, ns = parse_tei(p)
        root = merge_us(root)
        root = remove_dead_nextprev_links(root)
        write_tei(root, p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records_folder", type=str, default="corpus/records")
    parser.add_argument("-s", "--start", type=int, default=1867, help="Start year")
    parser.add_argument("-e", "--end", type=int, default=2022, help="End year")
    args = parser.parse_args()
    main(args)
