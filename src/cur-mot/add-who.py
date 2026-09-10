#!/usr/bin/env python3
"""
Add `who=` attribute to signators.
"""
from common.xml_utils import write_xml
from glob import glob
from lxml import etree
from pyriksdagen.utils import (
    parse_protocol
)
from tqdm import tqdm
import argparse, os, sys




def get_person_D(root, ns):
    D = {}
    person_list = root.find(f".//{ns['tei_ns']}listPerson")
    for p in person_list:
        try:
            _name = p.find(f"{ns['tei_ns']}name").text.strip()
        except:
            _name = None
        try:
            idno = p.find(f"{ns['tei_ns']}idno").text.strip()
        except:
            idno = None
        if idno is not None and _name is not None:
            D[idno] = _name
    return D


def match_ids(root, ns, person_D, unmatched, start, end):
    try:
        sig_list = root.find(f".//{ns['tei_ns']}div[@type='motSignatures']/{ns['tei_ns']}list")
        assert sig_list is not None
    except:
        try:
            sig_list = root.find(f".//{ns['tei_ns']}div[@type='motSignatures']/list")
            assert sig_list is not Non
        except:
            sig_list = root.find(f".//div[@type='motSignatures']/list")

    if sig_list is None:
        print(root.find(f".//{ns['tei_ns']}div[@type='motBody']"))
        utgone = False
        try:
            body = root.find(f".//{ns['tei_ns']}div[@type='motBody']").itertext()
        except:
            body = None
        if body is not None:
            for txt in body:
                print('-', txt.strip())
                if "Motionen utgår" in txt.strip():
                    print("x", body)
                    utgone = True
            if utgone == False:
                xmlns = ns['xml_ns']
                with open(f"riksdagen-motions/_no-sig-block_{start}-{end}.txt", "a+") as log:
                    log.write(f"{root.attrib[xmlns+'id']}\n")
    else:
        for k, v in person_D.items():
            found = False
            for s in sig_list:
                if v in s.text.strip():
                    s.attrib["who"] = k
                    found = True
            if not found:
                found_2 = []
                for s in sig_list:
                    X = v.split()
                    matches = []
                    for x in X:
                        if x in s.text.strip():
                            matches.append(True)
                    if len(matches)/len(X) >= 0.5:
                        found_2.append((len(matches)/len(X), (k,v)))
                if len(found_2) > 0:
                    if len(found_2) == 1:
                        s.attrib["who"] = found_2[0][1][0]
                        found = True
                if not found:
                    unmatched += 1
                    print(v, [s.text.strip() for s in sig_list])

    return root, unmatched




def main(args):
    data_location = os.environ.get("MOTIONS_PATH", "data")
    motions = glob(f"{data_location}/*/*.xml")
    motions = [m for m in motions if \
        m.split("/")[-2][:4] >= args.start and\
        m.split("/")[-2][:2]+m.split("/")[-2][-2:] < args.end]
    unmatched = 0
    for mot in tqdm(motions):
        print("~~", mot)
        root, ns = parse_protocol(mot, get_ns=True)
        person_D = get_person_D(root, ns)
        root, unmatched = match_ids(root, ns, person_D, unmatched, args.start, args.end)
        write_xml(root, mot)

    print(unmatched)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--start", required=True)
    parser.add_argument("-e", "--end", required=True)
    args = parser.parse_args()
    main(args)
