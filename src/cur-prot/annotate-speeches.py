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




logger = get_logger(name="Annotate Speeches")




def find_speeches(root, ns, record):
    speeches = {}
    speech_elems = []
    prev_intro_id, current_intro_id = None, None
    passed_intro = False
    def add_to_speeches(speeches, speech_elems, id1, id2):
        """
        Generate a deterministic speech ID and add to the dict.

        The ID is generated from a deterministic seed, which
        consists of "speech" + the ID of the speaker introduction
        before + the ID of the speaker introduction after. If no
        speaker introduction is found after due to the div
        changing, "div" is added instead. If no introduction is
        found due to the document ending, nothing is added at the
        end.
        """
        seed = "speech" + id1
        if id2 is not None:
            seed += id2

        speech_ID = get_formatted_uuid(seed=seed)
        speeches[speech_ID] = speech_elems
        return speeches

    TEI_NS = ns['tei_ns']
    body = root.findall(f".//{TEI_NS}body")
    if not len(body) == 1:
        logger.error(f"Body elem != 1 in {record} | Body elems: {body}")

    try:
        body = body[0]
    except:
        return speeches
    for elem in body.iter():
        # Remove namespace from the tag
        tag = elem.tag.split("}")[-1]
        if "div" == tag:
            if len(speech_elems) > 0:
                speeches = add_to_speeches(speeches, speech_elems, current_intro_id, "div")
                speech_elems = []
            prev_intro_id, current_intro_id = None, None
            passed_intro = False

        elif tag == "note" and elem.attrib.get("type") == "speaker":
            #print("passed intro")
            passed_intro = True

            prev_intro_id = current_intro_id
            current_intro_id = elem.get(f"{ns['xml_ns']}id")
            if len(speech_elems) > 0:
                speeches = add_to_speeches(speeches, speech_elems, prev_intro_id, current_intro_id)
                speech_elems = []
        elif tag == "u" and passed_intro:
            speech_elems.append(elem.get(f"{ns['xml_ns']}id"))

    if len(speech_elems) > 0:
        speeches = add_to_speeches(speeches, speech_elems, current_intro_id, None)

    if not len(list(speeches.keys())) == len(list(set(list(speeches.keys())))):
        raise ValueError(f"You probably have a duplicate UUID,")
    return speeches


def add_speeches_to_metadata(speeches, root, ns):
    """
    Add speeches and text-containing sub elements to the TEIHeader
    under the constitution element.
    """
    teiHeader = root.find(f"{ns['tei_ns']}teiHeader")
    if teiHeader is None:
        raise ValueError(f"No TEI header found")

    composition = None
    constitution = teiHeader.find(f".//{ns['tei_ns']}constitution")
    if constitution is None:
        logger.debug("constitution element not found.")
        profileDesc = teiHeader.find(f"{ns['tei_ns']}profileDesc")
        if profileDesc is None:
            logger.debug("Creating profileDesc elem")
            profileDesc = etree.SubElement(teiHeader, "profileDesc")
        textDesc = profileDesc.find(f"{ns['tei_ns']}textDesc")
        if textDesc is None:
            logger.debug("Creating textDesc elem")
            textDesc = etree.SubElement(profileDesc, "textDesc")
            channel = etree.SubElement(textDesc, "channel")
            channel.set("mode", "s")
        composition = textDesc.find(f"{ns['tei_ns']}composition")
        if composition is None:
            logger.debug("Creating composition elem")
            composition = etree.SubElement(textDesc, "composition")

    if composition is None:
        textDesc = teiHeader.find(f".//{ns['tei_ns']}textDesc")
        composition = textDesc.find(f".//{ns['tei_ns']}constitution")
        for note in list(composition):
            composition.remove(note)

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
        speeches = find_speeches(root, ns, record)
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
