#!/usr/bin/env python3
"""
Curate digital-original motions from json --> tei xml
"""
from common.xml_utils import write_xml
from common.metadata import (
    fetch_template,
    populate_correspDesc,
    populate_person_list,
    populate_textClass,
    prepare_uppgift,
    prepare_referens,
    set_source_desc,
    set_title_stmt,
)
from glob import glob
from lxml import etree
from common.html_parser import populate_from_html
from pyriksdagen.match_mp import (
    clean_names,
    match_mp,
    name_equals,
    name_almost_equals,
    names_in,
)
from pyriksdagen.metadata import load_Corpus_metadata
from pyriksdagen.utils import (
    get_data_location,
    get_formatted_uuid,
    parse_protocol,
)
from tqdm import tqdm
import argparse, json, os
import pandas as pd
import sys



def main(args):
    if args.metadata_folder is None:
        metadata_folder = os.environ.get("METADATA_PATH", "data")
    else:
        metadata_folder = args.metadata_folder

    if args.recompile_metadata:
        db = load_Corpus_metadata(metadata_folder = metadata_folder)
        db.to_pickle(args.metadata_path)
    else:
        db = load_Corpus_metadata(read_db_from=args.metadata_path)

    db.rename(columns={"person_id":"id"}, inplace=True)

    IDs = pd.read_csv(f"{metadata_folder}/external_identifiers.csv")
    person = pd.read_csv(f"{metadata_folder}/person.csv")
    party_affil = pd.read_csv(f"{metadata_folder}/party_affiliation.csv")

    if args.motions_repo is None:
        motions_xml = get_data_location('motions')
    else:
        motions_xml = args.motions_repo

    party_D = {
            "FP": "Q53764745",
            "fp": "Q53764745",
            "kds": "Q213654",
            "kd": "Q213654",
            "KD": "Q213654",
            "M": "Q110843",
            "m": "Q110843",
            "mp": "Q213451",
            "MP": "Q213451",
            "S": "Q105112",
            "s": "Q105112",
            "SD": "Q504069",
            "V": "Q110837",
            "v": "Q110837",
            "C": "Q110832",
            "c": "Q110832",
        }

    actions = dict(zip(
        ['Inlämnad', 'Granskad', 'Hänvisad', 'Bordlagd',
            'Utgången', 'Lagd till handlingarna', 'Förfallen', 'Återkallad',
            'Återtagen', 'Numrering', 'Inlämning', 'Överföring',
            'Utgår', 'Registrering', 'Hänvisning', 'Bordläggning',
            'Återkallande', 'Mottagning', 'Förfall', 'Granskning',
            'Hänvisningsförslag', 'Utskottsförslag', 'Avslutande av mottagning',
            ],
        ['Submitted', 'Reviewed', 'Referred', 'Dismissed',
            'Expired', 'Added to File', 'Expired', 'Revoked',
            'Retracted', 'Numbering', 'Submission', 'Transfer',
            'Expires', 'Registration', 'Referral', 'Dismissal',
            'Revoking', 'Reception', 'Expire', 'Reviewing',
            'Proposed_referral', 'Committee_proposal', 'Termination',
            ]))
    categories = dict(zip(
        ['Fristående motion', 'Följdmotion', 'Händelse', 'Händelse av större vikt'],
        ['fristående', 'följd', 'händelse', 'händelse']))
    statuses = dict(zip(
        ['Ärendet är avslutat',
            'Motionen bereds i utskott',
            'Motionen behandlas inte',
            'Motionen är inlämnad',
        ],
        [{"tx":'The case is closed', "kod":"closed"},
            {"tx": 'The motion is being prepared in committee', "kod": "committeeProcessing"},
            {"tx": 'The motion is not being processed', "kod": "dismissed"},
            {"tx": "The motion is submitted", "kod": "inlämnad"},
        ]))
    mot_types = {
        'Enskild motion':"enskild",
        'Kommittémotion':"kommitte",
        'Partimotion':"parti",
        "Flerpartimotion": "flerparti",
        "Fristående motion": "fristående",
        "Följdmotion": "följd"
    }

    skip = ["input/motions/mot-2010-2013.json/h002fi15.json"]

    if args.motion is not None:
        json_mots = [f"{args.input_dir}/{args.json_dir}/{args.motion}"]
    else:
        json_mots = glob(f"{args.input_dir}/{args.json_dir}/*.json")

    for json_mot in tqdm(json_mots[:]):
        if json_mot in skip:
            print("~~~skip")
            continue
        print(json_mot)
        with open(json_mot, 'r', encoding='utf-8-sig') as inf:
            J = json.load(inf)
            if J["dokumentstatus"]["dokument"]["status"] == "DokTextExtraktor":
                continue
                # Curated separately
                #    mot-2014-2017/h202pkd1.json
                #    mot-2014-2017/h202sd200.json
            py = J["dokumentstatus"]["dokument"]["rm"].replace("/", "")
            number = f'{J["dokumentstatus"]["dokument"]["nummer"]:0>5}'
            organ = J["dokumentstatus"]["dokument"]["organ"]
            if organ is None:
                organ = ''
            docid = f"mot-{py}-{organ}-{number}"
            if not os.path.exists(f"{motions_xml}/{py}"):
                os.mkdir(f"{motions_xml}/{py}")

            #print('\n\n\n\n')
            #print(docid, json_mot)

            # prepare doc variables
            dokument = J["dokumentstatus"]["dokument"]

            if dokument["subtyp"] is not None and dokument["subtyp"] != '':
                motType = dokument["subtyp"]
            else:
                motType = None

            if 'dokbilaga' in J["dokumentstatus"]:
                dokbilaga = J["dokumentstatus"]["dokbilaga"]
            else:
                dokbilaga = None

            if "dokintressent" in J["dokumentstatus"]:
                dokintressent = J["dokumentstatus"]["dokintressent"]
            else:
                dokintressent = None

            if "dokaktivitet" in J["dokumentstatus"] and \
                J["dokumentstatus"]["dokaktivitet"] is not None and \
                J["dokumentstatus"]["dokaktivitet"]["aktivitet"] is not None:
                if type(J["dokumentstatus"]["dokaktivitet"]["aktivitet"]) == dict:
                    aktivitet = [J["dokumentstatus"]["dokaktivitet"]["aktivitet"]]
                else:
                    aktivitet = J["dokumentstatus"]["dokaktivitet"]["aktivitet"]
            else:
                aktivitet = None

            uppgift = prepare_uppgift(J)
            if uppgift is not None and "Motionskategori" in uppgift:
                motCat = uppgift["Motionskategori"]["text"]
            else:
                motCat = None

            referens = prepare_referens(J)

            ##############
            # populate xml
            root, ns = fetch_template()
            root.attrib[f"{ns['xml_ns']}id"] = docid


            # header
            root = set_title_stmt(root, ns['tei_ns'],
                                  dokument["titel"],
                                  dokument["subtitel"])

            root = set_source_desc(root, ns['tei_ns'], dokument, dokbilaga)

            root, roles, party_D = populate_person_list(root, ns['tei_ns'], dokintressent,
                                               dokument['datum'].split(' ')[0],
                                               IDs, person, party_affil, db, party_D)

            root =  populate_correspDesc(root, ns, actions, roles, aktivitet,
                                              uppgift, statuses, referens)

            root = populate_textClass(root, ns['tei_ns'], motType,
                                      mot_types, motCat, categories)


            # body
            root = populate_from_html(root, ns, J["dokumentstatus"]["dokument"]["html"].strip(), py, json_mot)

            # write
            write_xml(root, f"{motions_xml}/{py}/{docid}.xml")

    #{print(k, ": ", v) for k, v in party_D.items()}




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input-dir",
                        type=str,
                        default="input/motions",
                        help="Base folder containing folders of json files from data.riksdagen.se. Default:input/motions")
    parser.add_argument("-j", "--json-dir",
                        type=str,
                        required=True,
                        help="e.g. mot-2022-2025.json")
    parser.add_argument("-r", "--motions-repo",
                        type=str,
                        default=None,
                        help="Path to motions XML repo, if unset, the script looks for an environment variable, then defaults to `data/`.")
    parser.add_argument("-m", "--motion",
                        type=str,
                        default=None,
                        help="curate a singleton motion from json. file should be located where -j points.")
    parser.add_argument("--metadata-folder", type=str, default=None)
    parser.add_argument("--recompile-metadata", action='store_true')
    parser.add_argument("--metadata-path", type=str, default="input/metadata/db.pkl")
    args = parser.parse_args()
    main(args)
