#!/usr/bin/env python3
"""
Split signature elements with multiple last names in them
"""
from lxml import etree
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from pyriksdagen.io import (
    parse_tei,
    write_tei,
)

from pyriksdagen.utils import (
    get_data_location,
    infer_metadata,
    first_and_last_names,
    get_formatted_uuid,
    XML_NS
)
from tqdm import tqdm
import polars as pl
from trainerlog import get_logger

LOGGER = get_logger("split-signatures")

def allcaps_convert(wd):
    """
    Convert allcaps words to 'Capitalized'
    
    Keep all other words the same
    """
    if len(wd) <= 1:
        return wd
    elif wd == wd.upper():
        return wd.lower().capitalize()
    else:
        return wd

def is_andersson_etc(s):
    """
    Heuristically detect -son ending last names
    """
    if len(s) == 0:
        return False
    else:
        return s[0] != s[0].lower() and s[-3:] == "son"

def is_party_label(s):
    """
    Detect party labels in brackets, such as (S) or (MP)
    """
    if len(s) < 3 or len(s) > 5:
        return False
    else:
        cond1 = s[0] == "("
        cond2 = s[-1] == ")"
        cond3 = s[1].isalpha()
        return cond1 and cond2 and cond3

def split_signature(elem, first_names, last_names, iort):
    t = ' '.join(elem.text.split())
    t = t.replace(" . ", " ")
    t_before = t
    words = t.split()

    new_parts = []
    current_part = [words[0]]
    for w0, w1 in zip(words[:-1], words[1:]):
        if "i" not in [w0,w1]:
            w0_clean = w0.replace(".", "")
            w1_clean = w1.replace(".", "")
            w0_clean = w0.replace(",", "")
            w1_clean = w1.replace(",", "")
            cond1 = allcaps_convert(w0_clean) in last_names.union(iort)
            cond1 = cond1 or is_andersson_etc(allcaps_convert(w0_clean)) or is_party_label(w0_clean)

            cond2 = allcaps_convert(w1_clean) in first_names
            cond2 = cond2 or (w1_clean != "" and w1_clean not in last_names and w1_clean[0].isupper())

            if cond1 and cond2:
                LOGGER.debug(f"Split between : {w0} and {w1}")
                t = t.replace(f"{w0} {w1}", f"{w0}; {w1}")
                new_parts.append(current_part)
                current_part = []

        current_part.append(w1)

    if len(current_part) >= 1:
        new_parts.append(current_part)

    new_parts = [" ".join(p) for p in new_parts]

    assert " ".join(new_parts) == t_before

    signature_list = elem.getparent()    
    original_id = elem.attrib[f"{XML_NS}id"]
    original_ix = signature_list.index(elem)

    if t_before != t:
        elem.attrib["who"] = "unknown"

        LOGGER.info(f"{t_before} => {t}")
        LOGGER.debug(f"{new_parts}")

        elem.text = new_parts[0]
        
        for ix, signature_text in enumerate(new_parts[1:]):
            item = etree.Element("item")
            item.text = signature_text
            id_seed = f"split_signature\n{original_id}\n{ix}\n{t_before}"
            item.attrib[f"{XML_NS}id"] = get_formatted_uuid(id_seed)

            letter_count = len([ch for ch in signature_text if ch.isalpha()])
            if letter_count >= 3:
                item.attrib["who"] = "unknown"
                item.attrib["type"] = "signature"
            signature_list.insert(original_ix + ix + 1, item)

        return len(new_parts) - 1

    return 0

#i-GVxPCrypLVRDCytv2cGoH3
def main(args):
    metadata_location = get_data_location("metadata")

    df_names = pl.read_csv(f'{metadata_location}/name.csv')
    df_iort = pl.read_csv(f'{metadata_location}/location_specifier.csv')
    first_names, last_names, iort = first_and_last_names(df_names, df_iort)

    # Ad-hoc first names
    #first_names = first_names + {"Erik", "Johan"}

    with open("first_names.txt", "w") as f:
        f.write("\n".join(list(first_names)))
    with open("last_names.txt", "w") as f:
        f.write("\n".join(list(last_names)))
    with open("iort.txt", "w") as f:
        f.write("\n".join(list(iort)))

    no_splits = 0
    # Pre-load regex for intro_to_dict for performance reasons
    for i, motion in enumerate(tqdm(sorted(args.motions))):
        py = motion.split("/")[2]
        if py in ["fort", "reg"]:
            continue
        root, ns = parse_tei(motion)

        # Filter db to only include MPs that had a mandate that year
        metadata = infer_metadata(motion)

        blocks = root.findall(f".//{ns['tei_ns']}signatureBlock")
        for signatureBlock in blocks:
            for item in signatureBlock.findall(f".//{ns['tei_ns']}item"):
                t = ' '.join(item.text.split())
                if len(t) > 0:
                    if item.attrib.get("type") == "signature":
                        wds = t.replace(".", "").split()
                        wds = t.replace(",", "").split()
                        wds = [allcaps_convert(wd) for wd in wds]
                        multiple_names = [wd in last_names or is_andersson_etc(wd) for wd in wds]
                        party_labels = [is_party_label(wd) for wd in wds]
                        has_multiple_names = sum(multiple_names) >= 2

                        # Check if there is any first names after the last surname or party label
                        last_name_index, party_label_index = None, None
                        if True in multiple_names:
                            last_name_index = len(wds) - multiple_names[::-1].index(True) - 1
                        if True in party_labels:
                            party_label_index = len(wds) - party_labels[::-1].index(True) - 1

                        max_ix = [ix for ix in [last_name_index, party_label_index] if ix is not None]
                        if len(max_ix) == 0:
                            max_ix = len(wds)
                        else:
                            max_ix = max(max_ix)
                        
                        tail = wds[max_ix:]
                        first_names_in_the_tail = [wd in first_names or (len(wd) == 1 and wd[0].isalpha() and wd[0] == wd[0].upper()) for wd in tail]
                        
                        if has_multiple_names or sum(party_labels) >= 2:
                            LOGGER.info(f"Multiple names: {t}")
                            no_splits += split_signature(item, first_names, last_names, iort)
                        elif sum(first_names_in_the_tail):
                            ix = first_names_in_the_tail.index(True)
                            wd = wds[ix + max_ix]
                            LOGGER.info(f"First name '{wd}' in funny position: {t}")
                            no_splits += split_signature(item, first_names, last_names, iort)

        write_tei(root, motion)

    LOGGER.train(f"In total, {no_splits} signature splits were made ({no_splits/len(args.motions)} per motion)")

if __name__ == '__main__':
    parser = fetch_parser("motions", docstring=__doc__)
    parser.add_argument("--redetect-knowns", action='store_true')
    parser.add_argument("--skip-multiple-names", action='store_true')
    parser.add_argument("--metadata-location",
                        type=str,
                        default="input/metadata/db.pkl",
                        help="path to compiled metadata db")
    parser.add_argument("--recompile-metadata", action='store_true')
    parser.add_argument("--write-compiled-db",
                        type=str,
                        default="True",
                        choices=["True", "False"],
                        help="write db after compile")

    args = parser.parse_args()
    LOGGER.info(f"args: {args}")
    main(impute_args(args))
