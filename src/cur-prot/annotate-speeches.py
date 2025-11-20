#!/usr/bin/env python3
"""
Create IDs for speeches. Add speech IDs to metadata.
"""
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




def find_speeches(root, ns):
    speeches = {}
    speech_elems = []
    passed_intro = False
    def add_to_speeches(speeches, speech_elems):
        seed = ''.join(speech_elems)
        speech_ID = get_formatted_uuid(seed=seed)
        speeches[speech_ID] = speech_elems
        return speeches
    for tag, elem in elem_iter(root):
        if tag == "note" and elem.attrib.get("type") == "speaker":
            #print("passed intro")
            passed_intro = True
            if len(speech_elems) > 0:
                speeches = add_to_speeches(speeches, speech_elems)
                speech_elems = []
        elif tag == "u" and passed_intro:
            speech_elems.append(elem.get(f"{ns['xml_ns']}id"))
    if len(speech_elems) > 0:
        speeches = add_to_speeches(speeches, speech_elems)

    try:
        assert len(list(speeches.keys())) == len(list(set(list(speeches.keys()))))
    except Exception as e:
        raise ValueError(f"You probably have a duplicate UUID, {e}")
    return speeches


def add_speeches_to_metadata(speeches, root, ns):
    teiHeader = root.find(f"{ns['tei_ns']}teiHeader")
    try:
        assert teiHeader is not None
    except Exception as e:
        raise ValueError(f"No TEI header found : {e}")
    try:
        constitution = teiHeader.find(f".//{ns['tei_ns']}constitution")
        assert constitution is not None
    except Exception as e:
        try:
            profileDesc = teiHeader.find(f"{ns['tei_ns']}")
            assert profileDesc is not None
            logger.debug("profileDesc elem found")
        except:
            logger.debug("Creating profileDesc elem")
            profileDesc = etree.SubElement(teiHeader, "profileDesc")
        try:
            textDesc = profileDesc.find(f"{ns['tei_ns']}textDesc")
            assert textDesc is not None
            logger.debug("textDesc elem found")
        except:
            logger.debug("Creating textDesc elem")
            textDesc = etree.SubElement(profileDesc, "textDesc")
            channel = etree.SubElement(textDesc, "channel")
            channel.set("mode", "s")
        try:
            composition = textDesc.find(f"{ns['tei_ns']}composition")
            assert composition is not None
            logger.debug("composition elem found")
        except:
            logger.debug("Creating composition elem")
            composition = etree.SubElement(textDesc, "composition")

    for id_, Us in speeches.items():
        speechNote = etree.SubElement(composition, "note")
        speechNote.set("type", "speech")
        speechNote.set(f"{ns['xml_ns']}id", id_)
        linkGrp = etree.SubElement(speechNote, "linkGrp")
        linkGrp.set("type", "u")
        for u in Us:
            ptr = etree.SubElement(linkGrp, "ptr")
            ptr.set("target", f"#{u}")
    return True




def main(args):
    for record in tqdm(args.records):
        logger.debug(record)
        root, ns = parse_tei(record)
        speeches = find_speeches(root, ns)
        if len(speeches) > 0:
            logger.debug(f"  {len(speeches)} speeches found")
            if add_speeches_to_metadata(speeches, root, ns):
                write_tei(root, record)
            else:
                logger.critical("Problem adding speeches to meatadata")
        else:
            logger.warn(f"No speeches found :: {record}")




if __name__ == '__main__':
    parser = fetch_parser("records", docstring=__doc__)
    main(impute_args(parser.parse_args()))
