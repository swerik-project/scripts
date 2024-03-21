#!/usr/bin/env python3
"""
plot records, speeches, and words per year
"""
#from lxml import etree
from pyriksdagen.utils import (
    elem_iter,
    get_data_location,
    parse_protocol,
    protocol_iterators,
)
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd




def count_words(t):
    tokens = [_.strip() for _ in t.split(' ') if len(_) > 0 and _ != '\n']
    return len(tokens)


def count_speeches(protocol):
    speeches, words= 0,0
    root, ns = parse_protocol(protocol, get_ns=True)
    for tag, elem in elem_iter(root):
        if tag in ["note"]:
            if 'type' in elem.attrib:
                if elem.attrib['type'] == 'speaker':
                    speeches += 1
        if tag == "u":
            for segelem in elem:
                words += count_words(segelem.text)
    Npb = len(root.findall(f".//{ns['tei_ns']}pb"))
    return speeches, words, Npb


def plot(df, out):
    df.set_index('year', inplace=True)
    fig, (ax1, ax2, ax3) = plt.subplots(3)#, sharex=True)
    plt.rcParams.update({'font.size': 14})
    ax1.plot(df['prot'])
    ax1.set_title("Records")
    ax1.set_ylim(bottom=0)
    ax2.plot(df['intros'])
    ax2.set_title("Speeches")
    ax2.set_ylim(bottom=0)
    scale_y2 = 1e3
    ticks_y2 = ticker.FuncFormatter(lambda x, pos: '{0:g}k'.format(x/scale_y2))
    ax2.yaxis.set_major_formatter(ticks_y2)
    ax3.plot(df['words'])
    scale_y = 1e6
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}M'.format(x/scale_y))
    ax3.yaxis.set_major_formatter(ticks_y)
    ax3.set_title("Words")
    ax3.set_ylim(bottom=0)
    for a in [ax1, ax2, ax3]:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    fig.set_size_inches(7,6)
    fig.tight_layout()
    plt.savefig(f"{out}/prot-intro-word.pdf", format="pdf", dpi=300)
    plt.show()




def main(args):
    # def dicts
    r = {}              # records
    s = {}              # speeches
    w = {}              # words
    p = {}              # pages
    dicts = [r, s, w, p]

    if args.records_path:
        records_path = args.records_path
    else:
        records_path = get_data_location('records')

    protocols = sorted(list(protocol_iterators(records_path)))
    print(f"Checking the {len(protocols)} protocols...")
    for prot in tqdm(protocols):
        year = prot.split('/')[-1].split('-')[1][:4]
        #print(prot, year)
        for d in dicts:
            if year not in d:
                d[year] = 0
        r[year] += 1
        Ns, Nw, Np = count_speeches(prot)
        s[year] += Ns
        w[year] += Nw
        p[year] += Np
    rows = []
    cols = ["year", "prot", "pages", "intros", "words", ]
    for k, v in r.items():
        rows.append([int(k), v, p[k], s[k], w[k]])
    df = pd.DataFrame(rows, columns = cols)
    print(df)
    print("\n\nPlotting")
    plot(df, args.output_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="input/plots/LREC")
    args = parser.parse_args()
    main(args)
