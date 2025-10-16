from lxml import etree
import argparse
from pyriksdagen.utils import elem_iter, protocol_iterators, get_formatted_uuid
from pyriksdagen.utils import TEI_NS, XML_NS
from pyriksdagen.io import parse_tei, write_tei
import tqdm 
import multiprocessing
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
import polars as pl
import string

def process_signature_block(text):
    text = text.replace(" i ", " i_")
    text = text.split()
    labels = [wd in surnames or "i_" in wd or is_party_abbrev(wd) for wd in text]
    next_lables = labels[1:] + [False]

    filtered_labels = [a and not b for a,b in zip(labels, next_lables)]

    # Process into a string with names delimited by a linebreak
    s = ""
    for wd, label in zip(text, filtered_labels):
        s += " " + wd
        if label:
            s += "\n"

    s = s.replace(" i_", " i ")
    return [name.strip() for name in s.split("\n") if len(name) >= 1], filtered_labels[-1]

def is_signature_heuristic(text):
    text = ''.join(filter(lambda c: c not in string.punctuation, text))
    name_matches = [name in surnames for name in text.split()]
    return sum(name_matches) >= 1

def process_motion(protocol):
    root, _ = parse_tei(protocol, get_ns=True)

    for body in root.findall(f".//{TEI_NS}body"):
        for div in body.findall(f".//{TEI_NS}div"):
            if div.attrib.get("type") == "signatureBlock":
                for l in div.findall(f".//{TEI_NS}list"):
                    if len(list(l)) >= 2:
                        
                        for elem in list(l):
                            text = "\n".join(elem.itertext())
                            text, last_is_surname = process_signature_block(text)
                            current = elem

                            if len(text) >= 2 and elem.attrib.get("who", "unknown") == "unknown":
                                # If the last name is not a surname, concatenate it to the next element
                                if not last_is_surname and l.index(elem) < len(l) - 1:
                                    last_name_in_list = text[-1]
                                    text = text[:-1]
                                    next_elem = l[l.index(elem)+1]
                                    next_elem.text = last_name_in_list + " " + next_elem.text

                                for t in text:
                                    p = etree.Element(f"{TEI_NS}item")
                                    p.text = t
                                    if is_signature_heuristic(t):
                                        p.attrib["type"] = "signature"
                                    print(t, is_signature_heuristic(t))
                                    l.insert(l.index(current)+1, p)
                                    current = p
                                
                                l.remove(elem)
                            print(text)

    write_tei(root, protocol)


def main(args):
    df_names = pl.read_csv("last_names.csv")
    surnames = set([name for name in df_names["name"] if len(name) >= 2])

    for motion in tqdm.tqdm(args.motions):
        process_motion(motion, surnames)


if __name__ == "__main__":
    parser = fetch_parser("motions")
    args = impute_args(parser.parse_args())
    main(args)

