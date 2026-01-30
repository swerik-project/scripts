#!/usr/bin/env python3
"""
generate an orderd word frequency list
"""
from pyriksdagen.utils import (
    get_data_location,
    parse_protocol,
    protocol_iterators,
)
from tqdm import tqdm
import argparse, json, re




def main(args):
    wf = {}
    punctuation = r"\[\]\{\}\(\)<>.,!§”$«»'\";"
    if args.records_path is None:
        args.records_path = get_data_location("records")
    records = sorted(list(protocol_iterators(args.records_path, start=args.start, end=args.end)))
    for record in tqdm(records):
        root, ns = parse_protocol(record, get_ns=True)
        segs = root.findall(f".//{ns['tei_ns']}seg")
        notes = root.findall(f".//{ns['tei_ns']}note")
        for elem_list in [notes, segs]:
            for elem in elem_list:
                lines = elem.text.splitlines()
                for line in lines:
                    words = [_.strip() for _ in line.split(' ') if _.strip() != '']
                    for word in words:
                    #    print(word)
                        word = word.lower()
                        word = re.sub(rf"[{punctuation}]", '', word)
                        word = re.sub(r'[\d\-\–\—/]+', '', word)
                        word = word.strip(":")
                    #    print("----->", word)
                        if word not in ["", '-', '–', '-', '—']:
                            if word not in wf:
                                wf[word] = 0
                            wf[word] += 1
    wf = dict(sorted(wf.items(), key=lambda item: item[1], reverse=True))
    with open(f"{args.output_path}{args.output_file}", "w+") as o:
        json.dump(wf, o, ensure_ascii=False, indent=4)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--start", type=int, default=None)
    parser.add_argument("-e", "--end", type=int, default=None)
    parser.add_argument("-r", "--records-path", type=str, default=None)
    parser.add_argument("-p", "--output-path", type=str, default="input/wf/")
    parser.add_argument("-o", "--output-file", type=str, default="wf.json")
    args = parser.parse_args()
    main(args)
