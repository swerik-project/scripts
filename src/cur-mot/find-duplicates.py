#!/usr/bin/env python3
from pyriksdagen.args import (
    fetch_parser,
    impute_args
)
from pyriksdagen.utils import parse_tei
from tqdm import tqdm
import shutil


def elements_equal(e1, e2):
    if e1.tag != e2.tag: return False
    if e1.text != e2.text: return False
    if e1.tail != e2.tail: return False
    if not e1.tag.endswith('pb'):
        if e1.attrib != e2.attrib: return False
    if len(e1) != len(e2): return False
    return all(elements_equal(c1, c2) for c1, c2 in zip(e1, e2))



def test_duplicates(new_file, list_, dry_run):
    #print(new_file, list_)
    root, ns = parse_tei(f"riksdagen-motions/{new_file}")
    new_file_body = root.find(f".//{ns['tei_ns']}div[@type='motBody']")
    #print(new_file_body)
    for file_ in list_:
        if file_ != new_file:
            root, ns = parse_tei(f"riksdagen-motions/{file_}")
            old_file_body = root.find(f".//{ns['tei_ns']}div[@type='motBody']")
    #        print("~~~", old_file_body)
            if elements_equal(new_file_body, old_file_body):
                print(f"Deleting {new_file} -- same as {file_}")
                if dry_run == False:
                    shutil.move(f"riksdagen-motions/{new_file}", "riksdagen-motions/_duplicates")
                break
            #else:
            #    print("OK")


def main(args):
    D = {}
    for motion in tqdm(args.motions):
        #print(motion)
        m, _, py, file_ = motion.split('/')
        if py not in D:
            D[py] = {}
        nr = file_[:-4].split('-')[-1]
        if nr not in D[py]:
            D[py][nr] = []
        D[py][nr].append(motion)

    with open("riksdagen-motions/_tmp.txt", 'r') as inf:
        new_files = inf.readlines()
        new_files = [_.strip() for _ in new_files if _.strip() != '']
    #print(new_files)

    for year, year_d in D.items():
        for nr, mots in year_d.items():
            mots = [_.replace("riksdagen-motions/", "") for _ in mots]
            for mot in mots:
                #print(mot)
                if mot in new_files:
                    test_duplicates(mot, mots, args.dry_run)




if __name__ == '__main__':
    parser = fetch_parser("motions")
    parser.add_argument("-n", "--dry-run", action='store_true')
    main(impute_args(parser.parse_args()))
