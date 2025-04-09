"""
Fix a common segmentation error: margin notes ending in '(Forts.)' are merged into the text body
"""
from lxml import etree
from pyriksdagen.utils import (
    get_data_location,
    parse_tei,
    protocol_iterators,
    write_protocol,
    TEI_NS,
    XML_NS,
    elem_iter
)
from tqdm import tqdm
import argparse
import re
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)


def split_text(t, pattern):
    s = t.split(pattern)
    p1, p2 = pattern.split()
    return s[0] + p1, p2 + s[1]

def forts_herr(root):
    #re.compile("\\(Forts.\\) Herr [] \\:")
    pattern = "(Forts.) Herr"
    for tag, elem in elem_iter(root):
        #if tag == "u":
        #    for seg in elem:
        #        if seg.text is not None:
        #            elemtext = " ".join(seg.text.split())
        #            if pattern in elemtext:
        #                #print(elemtext)
        #                t1, t2 = split_text(elemtext, pattern)
        #                seg.text = t1

        if tag == "note":
            if elem.text is not None:
                elemtext = " ".join(elem.text.split())
                if pattern in elemtext:
                    #print(elemtext)
                    t1, t2 = split_text(elemtext, pattern)
                    elem.text = t1
                    newchild = etree.Element(f"{TEI_NS}note")
                    newchild.text = t2
                    elem.addnext(newchild)


    return root


def ang_forts(root):

    exp = re.compile(r"Ang\. [\w ]{10,40}(.) \((f|F)orts(\.)\) ")
    #re.compile("\\(Forts.\\) Herr [] \\:")
    
    for tag, elem in elem_iter(root):
        if tag == "u":
            for seg in elem:
                if seg.text is not None:
                    elemtext = " ".join(seg.text.split())
                    if exp.match(elemtext) is not None:
                        matched_str = exp.match(elemtext).group(0)
                        if len(matched_str) * 1.05 < len(elemtext.strip()):

                            if elemtext.strip()[:4] == "Ang.":
                                elemtext = elemtext.replace(matched_str, "")
                                seg.text = matched_str
                                newchild = etree.Element(f"{TEI_NS}seg")
                                newchild.text = elemtext
                                seg.addnext(newchild)
        elif tag == "note":
            if elem.text is not None:
                elemtext = " ".join(elem.text.split())
                if exp.match(elemtext) is not None:
                    matched_str = exp.match(elemtext).group(0)
                    if len(matched_str) * 1.05 < len(elemtext.strip()):

                        if elemtext.strip()[:4] == "Ang.":
                            elemtext = elemtext.replace(matched_str, "")
                            elem.text = matched_str
                            newchild = etree.Element(f"{TEI_NS}note")
                            newchild.text = elemtext
                            elem.addnext(newchild)

    return root


def main(args):
    for protocol in tqdm(args.records):
        root, _ = parse_tei(protocol)
        if args.pattern == "forts-herr":
            root = forts_herr(root)
        else:
            root = ang_forts(root)
        write_protocol(root, protocol)


if __name__ == "__main__":
    parser = fetch_parser("records")
    parser.add_argument("--pattern", type=str, default="forts-herr", choices=['forts-herr', 'ang-forts'])
    main(impute_args(parser.parse_args()))
