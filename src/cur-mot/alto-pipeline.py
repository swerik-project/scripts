#!/usr/bin/env python3
"""
Curate motions from Alto Files
"""
from alto import parse_file, String

from common.xml_utils import (
    parse_xml,
    write_xml,
)
from glob import glob
from lxml import etree
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from pyriksdagen.motions.curate import (
    fetch_template,
)
from pyriksdagen.utils import get_data_location
from tqdm import tqdm
import argparse, os




def format_package(package):
    def _get_alts(committee):
        committee_D = {
                "a": ["AU", "au"],
                "bo": ["BoU"],
                "fi": ["FiU"],
                "fö": ["FöU"],
                "jo": ['JoU'],
                "ju": ['JuU', "juu"],
                "k": ['KU'],
                "kr": ['KrU', "kru"],
                "l": ['LU', "lu"],
                "n": ['NU', "nu"],
                "sf": ['SfU', 'sfu'],
                "sk": ['SkU', 'sku'],
                "so": ['SoU'],
                "t": ['TU', 'tu'],
                "u": ['UU', 'uu'],
                "ub":["UbU", "ubu"],
            }
        if committee in committee_D:
            return committee_D[committee]
        else:
            return []

    #print(package)
    spl = package.split('_')

    if len(spl) == 4:
        m, year, committee, number = spl
        if year == "199900":
            year = "19992000"
        return f"{m}-{year}-{committee}-{number:0>5}.xml", [f"{m}-{year}-{c}-{number:0>5}.xml" for c in _get_alts(committee.lower())]
    elif len(spl) == 6:
        m, year, season, chamber, committee, number = spl
        return f"{m}-{year}-{season}-{chamber}-{committee}-{number:0>5}.xml", [f"{m}-{year}-{season}-{chamber}-{c}-{number:0>5}.xml" for c in _get_alts(committee.lower())]
    elif len(spl) == 3:
        m, year, number = spl
        return f"{m}-{year}--{number:0>5}.xml", []
    else:
        raise ValueError(f"errmahgerd: {len(spl)}, {spl}")



def check_existing_tei(package, _path, no_make=False):
    _root, _, py, package_base, *rest = package.split("/")
    package, committee_alts = format_package(package_base)
    if py == "199900":
        py = "19992000"
    try:
        assert os.path.exists(f"{_path}/{py}/{package}")
        root, ns = parse_xml(f"{_path}/{py}/{package}", get_ns=True)
        return root, ns, f"{_path}/{py}/{package}"
    except:
        print("Checking alts:")
        for alt in committee_alts:
            print(f"  -- {_path}/{py}/{alt}")
            if os.path.exists(f"{_path}/{py}/{alt}"):
                print("      ¡OH! There it is.")
                root, ns = parse_xml(f"{_path}/{py}/{alt}")
                return root, ns, f"{_path}/{py}/{alt}"
        if int(py) > 1970:
            print("WARNING:", f"{_path}/{py}/{package} should exist, but I can't find it" )

    if no_make:
        return None, None, None
    else:
        root, ns = fetch_template()
        if not os.path.exists(f"{_path}/{py}"):
            os.mkdir(f"{_path}/{py}")
        return root, ns, f"{_path}/{py}/{package}"


def list_years(args):
    args = vars(args)
    #print(args)
    if args["parliament_year"] is not None:
        years = args["parliament_year"]
    elif args["start"] is not None:
        _range = [_ for _ in os.listdir(args["altopath"]) if os.path.isdir(f"{args['altopath']}/{_}") and _ not in ["fort", "reg"]]
        years = sorted([_ for _ in _range if args['start'] <= int(_[:4]) <= args['end']])
    else:
        raise Error("Gah... I don't know what to do! Did you set start/end, year or pass alto packages?")
    return years




def main(args):

    skip_packages = [
            "riksdagen-motions-alto/data/200405/mot_200405_MOT__20040/",    # Duplicate of motion with valid file name
            "riksdagen-motions-alto/data/200405/mot_200405_MOT__200405/",   # Duplicate of motion with valid file name
        ]

    if args.alto_packages is None or len(args.alto_packages) == 0:
        args.alto_packages = []
        years = list_years(args)
        for year in years:
            print("~~", type(year), years, year)
            args.alto_packages.extend(sorted(glob(f"{args.altopath}/{year}/*/")))

    print(args.alto_packages)
    for package in tqdm(args.alto_packages[:]):
        if package in skip_packages:
            continue
        print(package, "~")
        root, ns, outpath = check_existing_tei(package, args.data_folder, no_make=args.no_new_files)
        if root is None:
            print("skipping: unknown target")
            continue
        print(" ~~>", outpath, "~~")
        _body = root.find(f".//{ns['tei_ns']}div[@type=\"motBody\"]")
        if _body is None:
            _body = etree.SubElement(root.find(f".//{ns['tei_ns']}body"), "div")
            _body.attrib["type"] = "motBody"
        else:
            #print(_body)
            for child in list(_body):
            #print(" ", child, )
                _body.remove(child)

        body = etree.SubElement(_body, "div")


        alto_files = sorted(glob(f"{package}*.xml"))
        for ix, alto_file in enumerate(alto_files):
            print(f"    ----  {alto_file}  ----    ----    ----    ----")
            if ix == 0:
                pb = etree.Element("pb")
                _body.getparent().insert(0, pb)
            else:
                pb = etree.SubElement(body, "pb")
            #print("PB", pb, pb.getparent().tag, pb.getparent().attrib, alto_file)
            pb.attrib["facs"] = alto_file.replace("alto", "pdf")[:-4] + ".pdf"
            a = parse_file(alto_file)
            #print(alto_file)
            _blocks = a.extract_composed_blocks()
            for _block in _blocks:
                if _block is not None:
                    p = etree.SubElement(body, "p")
                    _block_text = [string.content
                                    for tb in _block.text_blocks
                                    for line in tb.text_lines
                                    for string in line.strings
                                    if isinstance(string, String)]
                    p.text = ' '.join(_block_text)
        write_xml(root, outpath)




if __name__ == '__main__':
    parser = fetch_parser("motions", docstring=__file__)
    parser.add_argument("--altopath",
                        type=str,
                        default=None,
                        help="Path to alto xml files for OCRed riksdagen motions. If not explicitely set, this defaults to an environment variable or `data/` if no suitable variable is found")
    parser.add_argument("--alto-packages",
                        type=str,
                        default=None,
                        nargs='*',
                        help="Pass specific alto packages to process. An alto package is a directory containing separate alto files for each specific physical page of a motion.")
    parser.add_argument("--no-new-files",
                      action='store_true',
                      help="don't make a new xml file under riksdagen-motions/ if you can't find the one corresponding to the alto package")
    args = parser.parse_args()
    if args.altopath is None:
        args.altopath = get_data_location("motions-alto")
    main(impute_args(args))
