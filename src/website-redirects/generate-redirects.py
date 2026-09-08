#!/usr/bin/env python3
"""
Generate redirects to pdf documents.
Assumptions:
 - Each repo of PDFs has a corresponding <doctype>-redirect (local name) with origin <doctype>-pdf in github.
 - redirects in the form of jekyll-friendly markdown redirects

Generate redirects (a) directly from the pdf files (b) a list of pdf files
"""
from git import Repo
from glob import glob
from tqdm import tqdm
from trainerlog import get_logger
import argparse, os, shutil




logger = get_logger(name="ebun")




def push_redirects(redirects, redirects_path):
    """
    Push redirects

    Args:
        redirects (list): list of redirects
        redirects_path (str): path
    """
    logger.info(redirects[:2])
    try:
        repo = Repo(redirects_path.replace("docs", ".git"))
        repo.index.add(redirects)
        repo.index.commit("feat:add redirects")
        origin = repo.remote(name='origin')
        origin.push("main")
        return True
    except Exception as e:
        logger.error(f"Some error: {e}")
        return False


def make_redirects(pdf_files, pdf_path, redirects_path):
    """
    Make a list of redirects from pdf files.

    Args:
        pdf_files (list): list of pdf diles
        pdf_path (str): path to pdf repo
        redirects_path (str): path to redirects repo
    """
     redirects = []
     for pdf in tqdm(pdf_files):
         file_ = f"{redirects_path}/docs/{pdf.split('data/')[1][:-4]}.md"
         os.makedirs(os.path.dirname(file_), exist_ok=True)
         with open(file_, "w+") as out:
             out.write(f"---\nlayout: default\nUpRedirect: https://pdf.swedeb.se/{pdf_path}/{pdf.split('data/')[1]}\n---\n")
         redirects.append(f"docs/{pdf[:-4]}.md")
     return redirects


def filter_by_year(files, years):
    """
    Filter files by years.

    Args:
        files (list): list of files
        years (list): list of years
    """
    filtered = []
    for f in files:
        fy = int(f.split("data/")[1].split('/')[0][:4])
        if fy in years:
            filtered.append(f)
    return filtered




def main(args):
    pdf_path = f"{args.doctype}-pdf"
    redirects_path = f"{args.doctype}-redirect"
    years = []

    if args.from_list is not None:
        with open(args.from_list, 'r') as inf:
            src_files = [_.strip() for _ in inf.readlines()]
    else:
        src_files = glob(f"{pdf_path}/data/**/*.pdf", recursive=True)

    if args.clobber_dest:
        children = os.listdir(f"{redirects_path}/docs")
        for child in children:
            if child != "_layouts":
                if os.path.isdir(f"{redirects_path}/docs/{child}"):
                    shutil.rmtree(f"{redirects_path}/docs/{child}")
                else:
                    os.remove(f"{redirects_path}/docs/{child}")

    if args.year is not None:
        years.append(args.year)
    if args.start is not None:
        [years.append(year) for year in list(range(args.start, args.end+1))]
    if len(years) > 0:
        src_files = filter_by_year(src_files, years)

    if not args.only_generate_list:
        redirects = make_redirects(src_files, pdf_path, redirects_path)

    if args.generate_list:
        with open(f"{redirects_path}/pdf-list.txt", 'w+') as outf:
            [outf.write(f"{f}\n") for f in src_files]

    if args.no_push == False:
        if push_redirects(redirects, redirects_path) == False:
            raise AssertionError("the push failed")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--clobber-dest",
                        action='store_true',
                        help="delete all current redirect MD files at the destination")
    parser.add_argument("-d", "--doctype",
                        choices=[
                                "riksdagen-records",
                                "riksdagen-motions",
                                "riksdagen-volumeG",
                                "valtiopaivat-records"
                            ],
                        required=True)
    parser.add_argument("-e", "--end",
                        type=int,
                        default=None)
    parser.add_argument("-f", "--from-list",
                        type=str,
                        default=None,
                        help="generate redirects from a list of files, otherwise, original pdfs")
    parser.add_argument("-g", "--generate-list",
                        action='store_true',
                        help="generate a list of pdf files from original pdf files")
    parser.add_argument("-G", "--only-generate-list",
                        action='store_true')
    parser.add_argument("-s", "--start",
                        type=int,
                        default=None)
    parser.add_argument("-y", "--year",
                        type=int,
                        default=None)
    parser.add_argument("--no-push", action='store_true')
    args = parser.parse_args()
    if args.from_list is not None and args.generate_list is True:
        raise ValueError("Set --from-list or --generate-list, but not both")
    if args.start is not None or args.end is not None:
        try:
            assert args.start is not None
            assert args.end is not None
        except Exception as e:
            logger.error("both --start and --end or neither\n{e}\nIf you only need one year, use --year")
    if args.clobber_dest and (args.year or args.start):
        raise ValueError("it's not smart to --clobber-dest on a subset of data")
    if args.only_generate_list:
        args.generate_list = True
        args.no_push = True
    main(args)

