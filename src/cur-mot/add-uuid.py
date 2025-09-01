#!/usr/bin/env python3
"""
Add deterministic UUIDs to elements that don't have them
"""
from common.xml_utils import (
    write_xml,
)
from glob import glob
from pyriksdagen.utils import (
    get_formatted_uuid,
    parse_tei,
)
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from tqdm import tqdm
import argparse, os



def add_to_seedtext(seedtext, elem):
    if elem.text is not None and elem.text.strip() != '':
        seedtext += elem.text.strip()
    elif elem.attrib is not None:
        seedtext += str(elem.attrib)
    else:
        seedtext += elem.tag
    return seedtext

def ids_are_unique(motions):
    print("Checking all generated IDs are unique...")
    IDs = []
    for mot in tqdm(motions):
        root, ns = parse_tei(mot)
        body = root.find(f".//{ns['tei_ns']}body")
        for _ in body.iter():
            if _.attrib is not None and f"{ns['xml_ns']}id" in _.attrib:
                IDs.append(_.attrib[f"{ns['xml_ns']}id"])
    print(len(IDs))
    try:
        assert len(IDs) == len(set(IDs))
        print("OK")
    except:
        print("\n\n\n\n\n\t\tNot OK\n\n\n\n\n")


def main(args):

    for mot in tqdm(args.motions):
        seedtext = os.path.basename(mot)
        root, ns = parse_tei(mot)
        body = root.find(f".//{ns['tei_ns']}body")
        for elem in body:
        #    print(elem.tag)
            for sub_1 in elem:
                for sub_2 in sub_1:
                    for sub_3 in sub_2:
        #                print(sub_3.tag)
                        seedtext = add_to_seedtext(seedtext, sub_3)
                        if sub_3.tag != f"{ns['tei_ns']}pb" and (sub_3.attrib is None or f"{ns['xml_ns']}id" not in sub_3.attrib):
                            sub_3.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
                    seedtext = add_to_seedtext(seedtext, sub_2)
                    if sub_2.tag != f"{ns['tei_ns']}pb" and (sub_2.attrib is None or f"{ns['xml_ns']}id" not in sub_2.attrib):
                        sub_2.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
                seedtext = add_to_seedtext(seedtext, sub_1)
                if sub_1.tag != f"{ns['tei_ns']}pb" and (sub_1.attrib is None or f"{ns['xml_ns']}id" not in sub_1.attrib):
                    sub_1.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
            seedtext = add_to_seedtext(seedtext, elem)
            if elem.tag != f"{ns['tei_ns']}pb" and (elem.attrib is None or f"{ns['xml_ns']}id" not in elem.attrib):
                elem.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)

        write_xml(root, mot)

    ids_are_unique(args.motions)



if __name__ == '__main__':
    parser = fetch_parser("motions")
    main(impute_args(parser.parse_args()))
