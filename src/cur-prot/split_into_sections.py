"""
Split protocol into <div> sections baseed on the paragraph sign '§' and other heuristics
"""
from lxml import etree
from pyriksdagen.utils import (
    get_data_location,
    infer_metadata,
    parse_protocol,
    protocol_iterators,
    write_protocol,
)
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd


TEI_NS = "{http://www.tei-c.org/ns/1.0}"
XML_NS = "{http://www.w3.org/XML/1998/namespace}"
skip = [ # These are empty // only teiHeader, no body
    "corpus/protocols/199192/prot-199192--001.xml",
    "corpus/protocols/199192/prot-199192--002.xml"
    ]




def clean_next_prev(div, DEBUG):
    if DEBUG: print("|-------- starting clean")
    Us = []
    for elem in div:
        if DEBUG: print("|", elem.tag)
        if elem.tag == f"{TEI_NS}u":
            Us.append(elem)
    if DEBUG: print("|", Us)
    if len(Us) > 0:
        if DEBUG: print("|", Us[0])
        if DEBUG: print("|", Us[-1])
        if 'prev' in  Us[0].attrib:
            del Us[0].attrib['prev']
        if 'next' in Us[-1].attrib:
            del Us[-1].attrib['next']
        if DEBUG: print("|-->", Us[0])
        if DEBUG: print("|-->", Us[-1])
    if DEBUG: print("|-------")
    return div


def create_divs(root, metadata, DEBUG):
    bodies = root.findall(f".//{TEI_NS}body")
    assert len(bodies) == 1
    body = bodies[0]
    if DEBUG: print("   --- 1 body")
    if metadata["chamber"] == "Första kammaren":
        div = body[0]
        div.attrib["type"] = "debateSection"
        return root
    if DEBUG: print("   --- Ret FK")
    old_divs = body.findall(f".//{TEI_NS}div")
    if DEBUG: print(old_divs)
    if DEBUG: print("   --- Found old fivs")
    current_div = etree.SubElement(body, f"{TEI_NS}div")
    for div in old_divs:
        for elem in div:
            if elem.text is not None:
                beginning = elem.text.strip()[:4]
                if "§" in beginning:
                    current_div = etree.SubElement(body, f"{TEI_NS}div")

            current_div.append(elem)
    if DEBUG: print("   --- ran through old divs")
    return root


def clean_divs(root, DEBUG):
    divs = list(root.findall(f".//{TEI_NS}body/{TEI_NS}div"))
    if DEBUG: print(f"Cleaning {len(divs)} divs")
    for div in divs:
        if DEBUG: print(div)
        if len(div) == 0:
            parent = div.getparent()
            parent.remove(div)
        else:
            div = clean_next_prev(div, DEBUG)
            if DEBUG: print("      --- Cleaned next prev")
            notes = div.findall(f".//{TEI_NS}note")
            intros = [n for n in notes if n.attrib.get("type") == "speaker"]
            if len(intros) >= 1:
                div.attrib["type"] = "debateSection"
            else:
                div.attrib["type"] = "commentSection"
    if DEBUG: print("   --- typed new divs")
    return root


def convert_u_heuristic(root):
    rows = []
    for div in list(root.findall(f".//{TEI_NS}div")) + list(root.findall(".//div")):
        for elem in div:
            if elem.attrib.get("type") == "speaker":
                break

            if elem.tag == f"{TEI_NS}u":
                protocol_id = root.findall(f".//{TEI_NS}text")[0].findall(f".//{TEI_NS}head")[0].text
                print(protocol_id, "utterance before intro", elem.attrib[f"{XML_NS}id"])
                for seg in elem:
                    rows.append([seg.attrib[f"{XML_NS}id"], "note"])
    return rows


def nextprev_clean(root, DEBUG):
    divs = list(root.findall(f".//{TEI_NS}body/{TEI_NS}div"))
    if DEBUG: print(f"Cleaning {len(divs)} divs")
    for div in divs:
        if DEBUG: print(div)
        div = clean_next_prev(div, DEBUG)
    return root


def flow(root, rows, metadata, DEBUG):
    if args.nextprev_only:
        if DEBUG: print("Only cleaning next/prev attribs")
        root = nextprev_clean(root, DEBUG)
    else:
        if DEBUG: print("(re)creating divs from scratch")
        root = create_divs(root, metadata, DEBUG)
        root = clean_divs(root, DEBUG)
        if DEBUG: print(" --- finished create_divs")
        protocol_rows = convert_u_heuristic(root)
        if DEBUG: print(" --- finished convert_u")
        rows = rows + protocol_rows
    return root, rows




def main(args):
    DEBUG = args.debug
    rows = []
    skip_counter = 0
    failures = []

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
        if DEBUG: print(protocol)
        if protocol in skip:
            if DEBUG: print("!!! SKIPPING")
            continue
        root = parse_protocol(protocol)
        metadata = infer_metadata(protocol)
        try:
            root, rows = flow(root, rows, metadata, DEBUG)
        except Exception:
            skip_counter += 1
            failures.append(protocol)
            print(f"Problem with {protocol} ... Skipping ...")
        else:
            write_protocol(root, protocol)

    if len(rows) > 0:
        df = pd.DataFrame(rows, columns=["id", "preds"])
        df.to_csv(args.outpath, index=False)

    print("FAILURES:", skip_counter)
    [print("~~>", _) for _ in failures]




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--start", type=int, default=1867, help="Start year")
    parser.add_argument("-e", "--end", type=int, default=2022, help="End year")
    parser.add_argument("-r", "--records-folder",
                        type=str,
                        default=None,
                        help="(optional) Path to records folder, defaults to environment var or `data/`")
    parser.add_argument("-p", "--protocol",
                        type=str,
                        default=None,
                        help="operate on a single protocol")
    parser.add_argument("-d", "--debug", action="store_true", help="Print debug statements")
    parser.add_argument("-c", "--nextprev-only",
                        action="store_true",
                        help="Only clean up next-prev attrs.")
    parser.add_argument("--outpath", type=str, default="input/segmentation/section_heuristic_preds.csv")
    args = parser.parse_args()
    main(args)
