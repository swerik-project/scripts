#!/usr/bin/env python3
from lxml import etree
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from pyriksdagen.io import (
    parse_tei,
    write_tei
)
from pyriksdagen.utils import (
    elem_iter,
    get_formatted_uuid,
)
from tqdm import tqdm
from trainerlog import get_logger
import os




logger = get_logger(name="Trainer Log", level=os.environ.get("LOGLEVEL", None))


def get_source(py):
    if py <= "199495":
        return "OCR"
    else:
        return "digital-origin"


def add_source(root, source, ns):
    try:
        profileDesc = root.find(f"{ns['tei_ns']}profileDesc")
        assert profileDesc is not None
        logger.debug("profileDesc elem found")
    except:
        logger.debug("Creating profileDesc elem")
        profileDesc = etree.SubElement(root.find(f"{ns['tei_ns']}teiHeader"), "profileDesc")

    textClass = etree.SubElement(profileDesc, "textClass")
    classCode = etree.SubElement(textClass, "classCode")
    classCode.text = source
    return root


def add_url(root, url, ns):
    bibl = root.find(f".//{ns['tei_ns']}sourceDesc/{ns['tei_ns']}bibl")
    try:
        assert bibl is not None
    except:
        logger.critical("No sourceDescr/bibl elem found.")
    else:
        idno = etree.SubElement(bibl, "idno")
        idno.set("type", "URI")
        idno.set("subtype", "PDF")
        idno.text = url
    return root




def main(args):
    url_base = "https://swerik-project.github.io/riksdagen-records-pdf"
    for record in tqdm(args.records):
        logger.debug(record)
        spl = record.split('/')
        py = spl[-2]
        record_base=spl[-1][:-4]
        url = f"{url_base}/{py}/{record_base}"
        source = get_source(py)
        root, ns = parse_tei(record)
        root = add_source(root, source, ns)
        root = add_url(root, url, ns)
        write_tei(root, record)




if __name__ == '__main__':
    parser = fetch_parser("records", docstring=__doc__)
    main(impute_args(parser.parse_args()))
