#!/usr/bin/env python3
"""
Do the whole post-`pipeline.py` curation in order.

    classify_intros
    resegment
    add_uuid
    find_dates
    reclassify
    add_uuid
    dollar_sign_replace
    fix_capitalized_dashes
    redetect
    split_into_sections
    add_uuid

"""
import argparse, os, subprocess, sys


SCR = os.environ.get("SCRIPTS", 'scripts/source')
PIP = os.environ.get("PIPENV")
CONDA = os.environ.get("CONDAENV")



def list_args(l, args):
    if args.protocol:
        l.extend(["-p", args.protocol])
    else:
        l.extend(["-s", args.start, "-e", args.end])
    return l


def classify_intros(args):
    print("\n\n\n Classifying Introductions \n\n\n")
    l = [args.condaenv, f"{SCR}/cur-prot/classify_intros.py"]
    l = list_args(l, args)
    l.append("--cuda")
    with subprocess.Popen(l, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        for line in process.stdout:
            print(line.decode('utf8'))


def resegment(args):
    print("\n\n\n Resegmenting \n\n\n")
    l = [args.pipenv, f"{SCR}/cur-prot/resegment.py"]
    l = list_args(l, args)
    with subprocess.Popen(l, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        for line in process.stdout:
            print(line.decode('utf8'))


def add_uuid(args):
    print("\n\n\n Add UUID \n\n\n")
    l = [args.pipenv, f"{SCR}/cur-prot/add_uuid.py"]
    l = list_args(l, args)
    with subprocess.Popen(l, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        for line in process.stdout:
            print(line.decode('utf8'))


def find_dates(args):
    print("\n\n\n Find Dates \n\n\n")
    l = [args.pipenv, f"{SCR}/cur-prot/find_dates.py"]
    l = list_args(l, args)
    with subprocess.Popen(l, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        for line in process.stdout:
            print(line.decode('utf8'))


def reclassify(args):
    print("\n\n\n Reclassifying Intros \n\n\n")
    l = [args.condaenv, f"{SCR}/cur-prot/reclassify.py"]
    l = list_args(l, args)
    with subprocess.Popen(l, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        for line in process.stdout:
            print(line.decode('utf8'))


def dollar_sign_replace(args):
    print("\n\n\n Dollar Sign Replace \n\n\n")
    l = [args.pipenv, f"{SCR}/cur-prot/dollar_sign_replace.py"]
    l = list_args(l, args)
    with subprocess.Popen(l, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        for line in process.stdout:
            print(line.decode('utf8'))


def fix_capitalized_dashes(args):
    print("\n\n\n Fix Capitalized Dashes \n\n\n")
    l = [args.pipenv, f"{SCR}/cur-prot/fix_capitalized_dashes.py"]
    l = list_args(l, args)
    with subprocess.Popen(l, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        for line in process.stdout:
            print(line.decode('utf8'))


def redetect(args):
    print("\n\n\n Redetect Speakers \n\n\n")
    l = [args.pipenv, f"{SCR}/cur-prot/redetect.py"]
    l = list_args(l, args)
    with subprocess.Popen(l, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        for line in process.stdout:
            print(line.decode('utf8'))


def split_into_sections(args):
    print("\n\n\n Split into sections \n\n\n")
    l = [args.pipenv, f"{SCR}/cur-prot/split_into_sections.py"]
    l = list_args(l, args)
    with subprocess.Popen(l, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        for line in process.stdout:
            print(line.decode('utf8'))


def add_uuid_to_divs(args):
    print("\n\n\n Add UUID to divs \n\n\n")
    l = [args.pipenv, f"{SCR}/cur-prot/add_uuid_to_divs.py"]
    l = list_args(l, args)
    with subprocess.Popen(l, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        for line in process.stdout:
            print(line.decode('utf8'))


def empty_subp():
    print("\n\n\n  \n\n\n")
    with subprocess.Popen([], stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        for line in process.stdout:
            print(line.decode('utf8'))




def main(args):
    classify_intros(args)
    resegment(args)
    add_uuid(args)
    find_dates(args)
    reclassify(args)
    add_uuid(args)
    dollar_sign_replace(args)
    fix_capitalized_dashes(args)
    redetect(args)
    split_into_sections(args)
    add_uuid(args)
    # to do -- update corpus docs




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--start", type=str, default="1867", help="Start year")
    parser.add_argument("-e", "--end", type=str, default="2022", help="End year")
    parser.add_argument("-r", "--records-folder",
                        type=str,
                        default=None,
                        help="(optional) Path to records folder, defaults to environment var or `data/`")
    parser.add_argument("-p", "--protocol",
                        type=str,
                        help="operate on a single protocol")
    parser.add_argument("--pipenv",
                        type=str,
                        default=None,
                        help="Path to pip env. If unset, looks for environment variable, else fails.")
    parser.add_argument("--condaenv",
                        type=str,
                        default=None,
                        help="Path to conda env. If unset, looks for environment variable, else fails.")
    args = parser.parse_args()
    if args.pipenv == None:
        if PIP == None:
            print("You need to set a pip env.")
            sys.exit()
        else:
            args.pipenv = PIP
    if args.condaenv == None:
        if CONDA == None:
            print("You need to set a pip env.")
            sys.exit()
        else:
            args.condaenv = CONDA
    main(args)
