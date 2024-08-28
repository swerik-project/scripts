"""
Map introductions to the speaker in the metadata.
"""
from lxml import etree
import argparse
from pyriksdagen.utils import protocol_iterators
from tqdm import tqdm
from pyriksdagen.utils import TEI_NS

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

def main(args):
    protocols = sorted(list(protocol_iterators(args.records_folder, start=args.start, end=args.end)))
    parser = etree.XMLParser(remove_blank_text=True)
    for p in tqdm(protocols):
        root = etree.parse(p, parser).getroot()
        root = merge_us(root)

        b = etree.tostring(
            root, pretty_print=True, encoding="utf-8", xml_declaration=True
        )
        with open(p, "wb") as f:
            f.write(b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records_folder", type=str, default="corpus/records")
    parser.add_argument("-s", "--start", type=int, default=1867, help="Start year")
    parser.add_argument("-e", "--end", type=int, default=2022, help="End year")
    args = parser.parse_args()
    main(args)
