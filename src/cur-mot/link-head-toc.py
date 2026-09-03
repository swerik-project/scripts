#!/usr/bin/env python3
"""
Link TOC entries with head elements
"""

from common.xml_utils import write_xml
from glob import glob
from lxml import etree
from pyriksdagen.utils import (
    parse_protocol
)
from tqdm import tqdm
import argparse, os


def main(main):
    data_location = os.environ.get("MOTIONS_PATH", "data")
    motions = glob(f"{data_location}/*/*.xml")
    motions = [m for m in motions if \
        m.split("/")[-2][:4] >= args.start and\
        m.split("/")[-2][:2]+m.split("/")[-2][-2:] < args.end]
    matched = 0
    total = 0
    for mot in tqdm(motions):
        root, ns = parse_protocol(mot, get_ns=True)
        toc = root.find(f".//{ns['tei_ns']}div[@type='TOC']/{ns['tei_ns']}list")
        if toc is not None:
            print(toc)
            heads = root.findall(f".//{ns['tei_ns']}head")
            for head in heads:
                total += 1
                match = False
                if head.text is not None and head.attrib is not None and f"{ns['xml_ns']}id" in head.attrib:
                    #print("  ", head.attrib)
                    for elem in toc:
                        if elem.text is not None:
                            if ' '.join([_.strip() for _ in head.text.split("\n")]) in ' '.join([_.strip() for _ in elem.text.split("\n")]):
                                elem.attrib["corresp"] = head.attrib[f"{ns['xml_ns']}id"]
                                match = True
                                matched += 1


                    print("  ", match, head.text.strip())
        write_xml(root, mot)
    print(matched, total, matched/total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--start", required=True)
    parser.add_argument("-e", "--end", required=True)
    args = parser.parse_args()
    main(args)

