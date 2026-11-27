#!/usr/bin/env python3
"""
Detect titles in motions.
"""
from lxml import etree
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from pyriksdagen.utils import (
    get_formatted_uuid,
    parse_tei,
    write_tei,
)
from tqdm import tqdm
import re



def add_post(elem, post_text):
    pe = etree.Element("p")
    pe.text = post_text
    pe.attrib["id"] = get_formatted_uuid()
    pe.attrib["type"] = "signatureBlock"
    elem.addnext(pe)

def main(args):
    noM = 0
    reallyNoM = 0
    bothM = 0
    oneM = 0
    multiM = 0
    date_pat = re.compile(r'(.+\.|^)(Stockholm|\S+\slän)\sden:?\s\d{1,2}\s((J|j)anuari|(F|f)ebruari|(M|m)ars|(A|a)pril|(M|m)aj|(J|j)uni|(J|j)uli|(A|a)ugusti|(S|s)eptember|(O|o)ktober|(N|n)ovember|(D|d)ecember)\s\d{4}(,|\.)?')

    for motion in tqdm(args.motions):
        m = 0
        root, ns = parse_tei(motion)
        meta_dates = root.findall(f".//{ns['tei_ns']}correspAction/{ns['tei_ns']}date")
        for elem in list(root.iter()):
            if elem.tag.endswith("}p"):
                if elem.text is not None: #and elem.text.strip().startswith('Stockholm den '):
                    text = ' '.join([_.strip() for _ in elem.text.splitlines() if _.strip() != ''])
                    #print(text)
                    date = date_pat.search(text)
                    if date:
                        #print("date --->", date.group(0))
                        m += 1
                        if date.group(0) == text:
                            elem.attrib["type"] = "date"
                        else:
                            pre = text[:date.start()].strip()
                            post = text[date.end():].strip()
                            if len(pre) > 0:
                                elem.text = pre
                                if "type" in elem.attrib and elem.attrib["type"] == "date":
                                    del elem.attrib["type"]
                                de = etree.Element("p")
                                de.text = date.group(0)
                                de.attrib["type"] = "date"
                                de.attrib["id"] = get_formatted_uuid()
                                if len(post) == 0:
                                    n = elem.getnext()
                                    if n is not None:
                                        n.attrib["type"] = "signatureBlock"
                                else:
                                    add_post(elem, post)
                                elem.addnext(de)

                            else:
                                elem.text = date.group(0)
                                if len(post) > 0:
                                    add_post(elem, post)
                                else:
                                    n = elem.getnext()
                                    if n:
                                        n.attrib["type"] = "signatureBlock"
        if m == 0:
            noM += 1
            if len(meta_dates) < 1:
                reallyNoM += 1
        else:
            if m == 1:
                oneM += 1
            else:
                multiM += 1
            if len(meta_dates) > 0:
                bothM += 1

        write_tei(root, motion)
        #print("________________________________")

    print(noM, oneM, multiM)
    print((noM+oneM+multiM) == len(args.motions), len(args.motions))
    print(reallyNoM, bothM)


if __name__ == '__main__':
    parser = fetch_parser("motions", docstring=__doc__)
    main(impute_args(parser.parse_args()))
