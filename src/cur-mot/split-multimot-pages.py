#!/usr/bin/env python3
"""
Chunk up multimotion files --> one motion per file.
"""
from common.args import (
    alto_args,
    list_years,
    verify_alto_args,
)
from common.xml_utils import (
    parse_xml,
    write_xml,
)
from glob import glob
from tqdm import tqdm
import argparse, os
import regex as re




def main(args):
    years = list_years(args)

    for year in years:
        print(year)
        p_count = 0
        nmf = 0
        no_match = 0
        match = 0
        multimatch = 0
        year_D = {}
        motions = sorted(glob(f"{args.motionspath}/{year}/*.xml"))
        for mot in tqdm(motions):
            #print(mot)
            mot_refs = []
            nr = int(mot.split('-')[-1].replace(".xml", ""))
            #print(nr)
            root, ns = parse_xml(mot, get_ns=True)
            pat = re.compile(fr'((N|n)r[\.\,]?\s{nr}[\.\,]?)')#{{i<=1,d<=1,s<=1,e<=1}}')
            ps = root.findall(f".//{ns['tei_ns']}p")
            m = False
            for  p in ps:
                p_count += 1
                _text = " ".join([_.strip() for _ in p.text.splitlines() if _.strip() != ''])
                ms = pat.match(_text)
                if ms is None:# or len(ms) == 0:
                    no_match += 1
                #elif len(ms) == 1:
                else:
                    match += 1
                    m = True
                    # do stuff here
                    #print(ms)
                #else:
                #    multimatch += 1
                #    [print(m) for m in ms]
            if not m:
                #print(mot)
                nmf += 1


            """
            fws = root.findall(f".//{ns['tei_ns']}fw")
            for fw in fws:
                m = pat.match(' '.join([_.strip() for _ in fw.text.splitlines() if _.strip() != '']))
                if m is not None:
                    _nr = m.group(6)
                    nr = _nr.strip()
                    is_range = False
                    is_B = False
                    for _ in "-–—":
                        if _ in nr:
                            is_range = True
                    if "B" in nr:
                        is_B = True
                        nr = nr.replace("B", "")
                        nr = nr.replace(" ", "")
                    if is_range:
                        #print([_.strip() for _ in re.split('\D', nr) if _.strip() != ''])
                        start, end = [_.strip() for _ in re.split('\D', nr) if _.strip() != '']
                        if len(list(range(int(start), int(end)+1))) > 5:
                            print(mot, len(list(range(int(start), int(end)+1))))
                        for x in range(int(start), int(end)+1):
                        #    #print("  ", x)
                            ref = (x, is_B)
                            if ref not in mot_refs:
                                mot_refs.append(ref)
                    else:
                        ref = (nr, is_B)
                        if ref not in mot_refs:
                            mot_refs.append(ref)
                else:
                    print("WARNING: No Match in fw elem")
                    no_match += 1
            #print(mot_refs)
            if len(mot_refs) > 1:
                c += 1
                year_D[mot] = mot_refs
            """
        print("  ", p_count)
        print("  ", nmf)
        print("  matching elems:", match)
        print("  non-matching elems:", no_match)
        print("  elems with multi matches:", multimatch)

        for k, v in year_D.items():
            print(k, v)
if __name__ == '__main__':
    parser = alto_args(__doc__)
    args = parser.parse_args()
    main(verify_alto_args(args))
