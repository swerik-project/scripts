#!/usr/bin/env python3
"""
OCR PDF files in a path. A path should be of the following format:

    <basename>/
    |_ <path>/
    |   |_ abc_1.pdf
    |   |_ abc_2.pdf

The output will:

- augment the pdf base path as follows

    <basename>/
    |_ <path>/
    |   |_ abc_1.pdf
    |   |_ abc_2.pdf
    |    |_ abc_1/
    |        |_ abc_1_0001.pdf
    |        |_ abc_1_0002.pdf
    |        |_ abc_1_0001.png
    |        |_ abc_1_0002.png
    |    |_ abc_2/
    |        |_ abc_2_0001.pdf
    |        |_ abc_2_0002.pdf
    |        |_ abc_2_0001.png
    |        |_ abc_2_0002.png

- and create alto xml files for each page

    <altopath>/
    |_ <path>/
    |    |_ abc_1/
    |        |_ abc_1_0001.xml
    |        |_ abc_1_0002.xml
    |    |_ abc_2/
    |        |_ abc_2_0001.xml
    |        |_ abc_2_0002.xml
"""
from glob import glob
from pypdf import PdfWriter, PdfReader
from pyriksdagen.utils import get_data_location
from tqdm import tqdm
from trainerlog import get_logger
import argparse, cv2
import numpy as np
import os, pytesseract, subprocess
import shutil




logger = get_logger("OCR")
envs = {
        "prot": "RECORDS",
        "mot": "MOTIONS"
    }
dflt_paths = {
        "prot": "records",
        "mot": "motions"
    }




def write(img, outpath):
    cv2.imwrite(outpath,img)


def get_grayscale(image, ipath, ibase):
    """
    convert image to greyscale
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    write(img, f"{ipath}{ibase}_1_grayscale.jpg")
    return img, ipath, ibase


def thresholding(image, ipath, ibase):
    img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    write(img, f"{ipath}{ibase}_2_thresholding.jpg")
    return img, ipath, ibase


def preprocess_img(img, ipath, ibase):
    """
    greyscale and threshold images
    (seems unnecessary with tesseract)
    """
    return thresholding(*get_grayscale(img, ipath, ibase))


def extract_imgs(pdf, to):
    imgs = glob(f"{to}/*.png")
    for img in imgs:
        os.remove(img)
    pages = [p for p in os.listdir(f"{to}") if p.endswith(".pdf")]
    for page in pages:
        try:
            ok_code = subprocess.run(["pdfimages", "-png", 
                                      f"{to}/{page}", f"{to}/{os.path.basename(page)[:-4]}"],
                                      #stdout=subprocess.PIPE,
                                      checked=True
                                      stderr=subprocess.STDOUT)#,
                                      #text=False)
        except subprocessCalledProcessError as e:
            logger.critical(f"extract imgs error: {e}")
            raise
    return to


def extract_pages(pdf, to):
    """
    paginate multipage pdfs
    """
    try:
        ok_code = subprocess.run(["gs", 
                                  #"-dDEBUG", 
                                  "-dNOPAUSE", 
                                  "-dNOPROMPT", 
                                  "-dBATCH",
                                  "-dSAFER",
                                  #"-dPDFSTOPONERROR",
                                  #"-dVERBOSE", 
                                  "-sDEVICE=pdfimage24",
                                  "-r300",
                                  f'-sOutputFile={to}/{os.path.basename(pdf)[:-4]}_%04d.pdf', 
                                  pdf], checked=True)#,
                             #stdout=subprocess.PIPE,
                             #stderr=subprocess.STDOUT)#,
                             #text=False)
    except subprocess.CalledProcessError as e:
        logger.critical(f"Extract pages error: {e}")
        raise
    else:
        logger.info(f"Pdf is paginated: {ok_code}")
    return extract_imgs(pdf, to)


def ocr(img, outpath):
    """
    perform OCR on an image
    """
    with open(outpath, 'wb+') as outf:
        outf.write(pytesseract.image_to_alto_xml(img, lang='swe'))




def main(args):

    if args.basename:
        basename = args.basename
    else:
        basename = os.environ.get(
            f"{envs[args.doc_type]}_PDF",
            f"riksdagen-{dflt_paths[args.doc_type]}-pdf/data")

    if ocr:
        if args.altopath:
            altopath = args.altopath
        else:
            altopath = os.environ.get(
                f"{envs[args.doc_type]}_ALTO",
                f"riksdagen-{dflt_paths[args.doc_type]}-alto/data")

    if args.docs:
        pdfs = [glob(f"{basename}/{args.path}/{_}")[0] for _ in args.docs]
    else:
        pdfs = glob(f"{basename}/{args.path}/*.pdf")


    for pidx, pdf in enumerate(pdfs, start=1):
        print(pidx, "of", len(pdfs), "::", pdf)
        if not os.path.exists(f"{os.path.dirname(pdf)}/{os.path.basename(pdf)[:-4]}"):
            os.mkdir(f"{os.path.dirname(pdf)}/{os.path.basename(pdf)[:-4]}")

        year = os.path.basename(os.path.dirname(pdf))
        doc_base = os.path.basename(pdf)[:-4]

        if ocr:
            logger.info("Checking alto path...")
            if not os.path.exists(f"{altopath}/{year}"):
                os.mkdir(f"{altopath}/{year}")
            if not os.path.exists(
                f"{altopath}/{year}/{doc_base}"):
                os.mkdir(f"{altopath}/{year}/{doc_base}")
            logger.info("... OK")

        if args.clobber_dest:
            logger.info("Clobbering alto path...")
            dcontent = glob(f"{altopath}/{year}/{doc_base}/*")
            for X in dcontent:
                try:
                    if os.path.isfile(X):
                        os.remove(X)
                    elif os.path.isdir(X):
                        shutil.rmtree(X)
                except Exception as e:
                    logger.warn(f"Failed to delete {X}:\n{e}")
            logger.info("... OK")

        if args.repair_pdf:
            logger.info("repairing pdf")
            ok_code = subprocess.run(["mutool", "clean", "-gg", pdf, pdf], check=True)
            logger.info(f"... done {ok_code}")

        if args.split_pdf:
            logger.info(f"Extracting pages from pdf {pidx} of {len(pdfs)} -- {pdf}")
            img_dir = extract_pages(pdf, f"{basename}/{year}/{doc_base}")
            logger.info(" --> OK")
        else:
            img_dir = f"{basename}/{year}/{doc_base}"

        if ocr:
            logger.info(f"Iterating through {len(img_dir)} images")
            img_dir_fs = glob(f"{img_dir}/*.png")
            for img_path in img_dir_fs:
                iidx = img_path.split('-')[-2]
                logger.info(f"  --> {img_path} -- pdf {pidx} / {len(pdfs)} : img {iidx} of {len(img_dir_fs)}")
                img_base = os.path.basename(img_path)[:-4]
                img = cv2.imread(f"{img_path}")
                if args.no_preprocessing:
                    logger.info("    skipping preprocessing")
                    alto_path = f"{altopath}/{year}/{doc_base}/{img_base}.xml"
                    logger.info("     --> starting OCR")
                    ocr(img, alto_path)
                    logger.info("        |-> done.")
                else:
                    logger.info("    preprocessing")
                    img, ipath, ibase = preprocess_img(img, img_dir, img_base)
                    logger.info("    |-> Done")
                    alto_path = f"{altopath}/{year}/{doc_base}/{img_base}.xml"
                    logger.info("     --> starting OCR")
                    ocr(img, alto_path)
                    logger.info("        |-> done.")




if __name__ == '__main__':
    def _str2bool(v):
        """
        converts true/false strings to actual bool value
        b/c argparse's type=bool behavior is absurd
        """
        if isinstance(v, bool):
            return v
        if v.lower() in ('true', '1', 't', 'y', 'yes'):
            return True
        elif v.lower() in ('false', '0', 'f', 'n', 'no'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-p', '--path',
                        required=True,
                        type=str,
                        help="Path of directory containing PDF files to OCR")
    parser.add_argument('-d', '--docs',
                        type=str,
                        nargs="+",
                        default=None,
                        help="List of specific docs, otherwise, all that match --doc-type in the path will be OCRed")
    parser.add_argument('-t', '--doc-type',
                        type=str,
                        #required=True,
                        help="Type of doc to OCR, e.g. mot, prop, prot.")
    parser.add_argument('-b', '--basename',
                        type=str,
                        help="Base path to pdf repo. If unset, looks for environment variable based on the doc-type, otherwise defualts to, e.g. riksdagen-records-pdf/data")
    parser.add_argument("-a", "--altopath",
                        type=str,
                        help="Path to alto files repo. If unset, looks for environment variable based on the doc-type, otherwise defualts to, e.g. riksdagen-records-alto/data")
    parser.add_argument("-s", "--split-pdf",
                        action='store_true',
                        help="Split up multipage pdfs --> one page per file.")
    parser.add_argument('-P', '--no-preprocessing',
                        type=bool,
                        default=True,
                        help="skip preprocessing steps")
    parser.add_argument("-c", "--clobber-dest",
                        action='store_true', 
                        help="remove everything from the destination directory before ocr")
    parser.add_argument("-r", "--repair-pdf", type=_str2bool, default=True)
    parser.add_argument("--ocr", type=_str2bool, default=True)
    args = parser.parse_args()
    main(args)
