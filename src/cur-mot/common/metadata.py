"""
Get the metadata into TEI
"""
from lxml import etree
from pyriksdagen.utils import (
    get_formatted_uuid,
    parse_protocol,
)
import pandas as pd

def fetch_template():
    root, ns = parse_protocol("input/motions/mot-template.xml", get_ns=True)
    return root, ns


def populate_correspDesc(root, ns, actions, roles, aktivitet, uppgift, statuses, referens):
    xml = ns['xml_ns']
    ns = ns['tei_ns']
    try:
        title = root.find(f".//{ns}titleStmt/{ns}titlr")
        assert title is not None
    except:
        try:
            title = root.find(f".//{ns}titleStmt/title")
        except:
            title = None
    try:
        author = root.find(f".//{ns}titleStmt/{ns}author")
        assert author is not None
    except:
        try:
            author = root.find(f".//{ns}titleStmt/author")
        except:
            author = None

    seedtext = root.attrib[f"{xml}id"]
    if title is not None and title.text is not None:
        seedtext = seedtext + '\n' + title.text
    if author is not None and author.text is not None:
        seedtext = seedtext + '\n' + author.text

    correspDesc = root.find(f".//{ns}correspDesc")
    if uppgift is not None and "statustext" in uppgift:
        status = etree.SubElement(correspDesc, "correspAction")
        status.attrib["type"] = "status"
        status.attrib["subtype"] = statuses[uppgift["statustext"]["text"]]["kod"]
        seedtext = seedtext + '\n' + uppgift["statustext"]["systemdatum"]

    if uppgift is not None and "Motionsgrund" in uppgift and \
        uppgift["Motionsgrund"]["text"] is not None:
        basis = etree.SubElement(correspDesc, "correspAction")
        basis.attrib["type"] = "basedOn"
        if "systemdatum" in uppgift["Motionsgrund"]:
            seedtext = seedtext + '\n' + uppgift["Motionsgrund"]["systemdatum"]
        else:
            seedtext = seedtext + '\n' + uppgift["Motionsgrund"]["text"]
        bas_id = get_formatted_uuid(seed=seedtext)
        basis.attrib[f"{xml}id"] = bas_id
        basisC = etree.SubElement(correspDesc, "correspContext")
        basisC.attrib["corresp"] = bas_id
        bas_ref = etree.SubElement(basisC, "ref")
        bas_ref.text = uppgift["Motionsgrund"]["text"]

    signed = etree.SubElement(correspDesc, "correspAction")
    seedtext = seedtext + '\n' + '|'.join([_['id'] for _ in roles])
    signed_id = get_formatted_uuid(seed=seedtext)
    signed.attrib[f"{xml}id"] = signed_id
    signed.attrib["type"] = "signed"
    signatories = False
    signed_context = etree.Element("correspContext")
    signed_context.attrib["corresp"] = signed_id
    for role in roles:
        if role['role'] == 'undertecknare':
            signatories = True
            snote = etree.SubElement(signed_context, "note")
            snote.attrib["type"] = "signatory"
            snote.attrib["corresp"] = role["id"]
    if signatories == True:
        correspDesc.append(signed_context)
    if aktivitet is not None:
        for a in aktivitet:
            ca = etree.SubElement(correspDesc, "correspAction")
            ca.attrib["type"] = actions[a['namn']].lower()
            datum = a['datum']
            seedtext = seedtext + '\n' + datum
            ca.attrib[f"{xml}id"] = get_formatted_uuid(seed=seedtext)
            date, time = datum.split(' ')
            date_elem = etree.SubElement(ca, "date")
            date_elem.attrib["when"] = date
            if time != "00:00:00":
                time_elem = etree.SubElement(ca, "time")
                time_elem.attrib["when"] = time

    if uppgift is not None and "Tilldelat" in uppgift and \
        uppgift["Tilldelat"]["text"] is not None:
        assigned = etree.SubElement(correspDesc, "correspAction")
        assigned.attrib["type"] = "assigned"
        ass_to = uppgift["Tilldelat"]["text"]
        if "systemdatum" in uppgift["Tilldelat"]:
            dt = uppgift["Tilldelat"]["systemdatum"]
            seedtext = seedtext + '\n' + dt
            date, time = dt.split(' ')
            ass_date = etree.SubElement(assigned, "date")
            ass_date.attrib["when"] = date
            if time != "00:00:00":
                ass_time = etree.SubElement(assigned, "time")
                ass_time.attrib["when"] = time
        else:
            seedtext = seedtext + '\n' + ass_to
        ass_id = get_formatted_uuid(seed=seedtext)
        assigned.attrib[f"{xml}id"] = ass_id
        ass_context = etree.SubElement(correspDesc, "correspContext")
        ass_context.attrib["corresp"] = ass_id
        assignee = etree.SubElement(ass_context, "ref")
        assignee.attrib["type"] = "assignedTo"
        assignee.text = ass_to

    if referens is not None:
        for r in referens:
            _a = etree.SubElement(correspDesc, "correspAction")
            _a.attrib["type"] = r["referenstyp"]
            seedtext = seedtext + '\n' + r['uppgift']
            _aID = get_formatted_uuid(seed=seedtext)
            _a.attrib[f"{xml}id"] = _aID
            _c = etree.SubElement(correspDesc, "correspContext")
            _c.attrib["corresp"] = _aID
            for k, v in r.items():
                if k != "referenstyp" and v is not None:
                    _ = etree.SubElement(_c, "ref")
                    _.attrib["type"] = k
                    _.text = v

    if uppgift is not None and "Granskninstext" in uppgift:
        review = etree.SubElement(correspDesc, "correspAction")
        review.attrib["type"] = "review"
        seedtext = seedtext + '\n' + uppgift["Granskninstext"]["systemdatum"]
        rev_id = get_formatted_uuid(seed=seedtext)
        revC = etree.SubElement(correspDesc, "correspContext")
        revC.attrib["corresp"] = rev_id
        revC_note = etree.SubElement(revC, "note")
        note.text = uppgift["Granskninstext"]["text"]
    return root


def populate_person_list(root, ns, dokintressent, docdate,
                         IDs, person, party_affil, db, party_D):

    def _try_get_party_id(pid, docdate, party_affil):
        party_id = None
        df = party_affil.loc[party_affil['person_id']==pid].copy()
        persparty = df['party_id'].unique()
        if len(persparty) == 1:
            party_id = persparty[0]
        elif len(persparty) > 1:
            persparty = df.loc[(df['start'] < docdate) & (df['end'] > docdate), 'party_id'].unique()
            if len(persparty) == 1:
                party_id = persparty[0]
        return party_id

    def _try_match_name(_name, abbrev, docdate, db):
        _name = _name.strip()
        d = {"name": clean_names(_name)}
        if abbrev is not None:
            d["party_abbrev"] = abbrev
        return match_mp(d, db,
                    [k for k,v in d.items()],
                    [name_equals, name_almost_equals, names_in])

    def _swericize_id(rdid, _name, IDs, person, party_affil, docdate, db, partibet):
        swerik = None
        gender = None
        party_id = None
        if rdid == "0":
            swerik = _try_match_name(_name, partibet, docdate, db)
        else:
            pIDs = IDs.loc[IDs['identifier'] == rdid]
            if len(pIDs) > 0:
                ids = pIDs['person_id'].unique()
                if len(ids) == 1:
                    swerik = ids[0]
        if swerik is not None:
            pp = person.loc[
                    (person["person_id"] == swerik) &
                    (pd.notnull(person['gender']))]
            g = pp['gender'].unique()
            if len(g) == 1:
                gender = g[0]
            if gender is None or gender == '':
                gender = "unknown"
            party_id = _try_get_party_id(swerik, docdate, party_affil)
        else:
            swerik = f"r-{rdid}"
        return swerik, gender, party_id
    roles = []
    try:
        author = root.find(f".//{ns}titleStmt/{ns}author")
        assert author is not None
    except:
        try:
            author = root.find(f".//{ns}titleStmt/author")
        except:
            author = None
    if dokintressent is not None:
        if type(dokintressent['intressent']) == dict:
            intressent = [dokintressent['intressent']]
        else:
            intressent = dokintressent['intressent']
        listPerson = root.find(f".//{ns}listPerson")
        for i in intressent:
            d = {}
            _ = etree.SubElement(listPerson, "person")

            _id, gender, party_id = _swericize_id(i["intressent_id"], i['namn'], IDs, person,
                                                 party_affil, docdate, db, i['partibet'])
            d["id"] = _id
            if gender is not None:
                _.attrib["gender"] = gender
            idno = etree.SubElement(_, "idno")
            idno.text = _id
            _name = etree.SubElement(_, "name")
            _name.text = i["namn"]
            d["name"] = i["namn"]
            if party_id is None:
                if i['partibet'] in party_D and party_D[i['partibet']] != "unknown":
                    party_id = party_D[i['partibet']]
                else:
                    party_D[i['partibet']] = "unknown"

            state = etree.SubElement(_, "state")
            state.attrib["type"] = "partyAffiliation"
            if party_id is not None:
                state.attrib["ref"] = party_id
            desc = etree.SubElement(state, "desc")
            desc.text = i["partibet"]
            d["role"] = i["roll"]
            if author is not None:
                if i["namn"] in author.text:
                    if "corresp" in author.attrib:
                        author.attrib["corresp"] = ' '.join([author.attrib["corresp"], _id])
                    else:
                        author.attrib["corresp"] = _id
            roles.append(d)
    return root, roles, party_D


def populate_textClass(root, ns, doksubtyp, mot_types, motCat, categories):
    textClass = etree.SubElement(root.find(f".//{ns}profileDesc"), "textClass")
    docType = etree.SubElement(textClass, "catRef")
    docType.attrib["scheme"] = "#docType"
    docType.attrib["target"] = "#mot"
    if doksubtyp is not None and doksubtyp != '-':
        motType = etree.SubElement(textClass, "catRef")
        motType.attrib["scheme"] = "#motionType"
        motType.attrib["target"] = f"#{mot_types[doksubtyp]}"
    if motCat is not None and motCat != '-':
        mot_cat = etree.SubElement(textClass, "catRef")
        mot_cat.attrib["scheme"] = "#motionType"
        mot_cat.attrib["target"] = f"#{categories[motCat]}"
    return root


def prepare_uppgift(J):
    names = ['Motionskategori', 'Tilldelat', 'statustext', 'Motionsgrund', 'Granskningstext']
    if "dokuppgift" in J["dokumentstatus"] and \
        J["dokumentstatus"]["dokuppgift"] is not None:
        if "uppgift" in J["dokumentstatus"]["dokuppgift"]:
            uppgift = J["dokumentstatus"]["dokuppgift"]["uppgift"]
        else:
            uppgift = None
    else:
        uppgift = None
    if uppgift is not None:
        if type(uppgift) == dict:
            uppgift = [uppgift]
        d = {}
        for u in uppgift:
            if u["namn"] not in names:
                print(f"unknown uppgift type {u['namn']}")
                print("Should be one of:", names)
                sys.exit()
            d[u["namn"]] = u
        return d
    else:
        return None


def prepare_referens(J):
    referenstyppor = ['behandlas_i']
    if "dokreferens" in J["dokumentstatus"] and \
        J["dokumentstatus"]["dokreferens"] is not None:
        referens = J["dokumentstatus"]["dokreferens"]["referens"]
        if type(referens) == dict:
            referens = [referens]
        for r in referens:
            if r["referenstyp"] not in referenstyppor:
                print(f"unknown dokreferens type: {r['referenstyp']}")
                print("Should be one of:", referenstyppor)
                sys.exit()
    else:
        referens = None
    return referens


def set_source_desc(root, ns, doc, attachments):
    bibl_elem = root.find(f".//{ns}bibl")
    if "typrubrik" in doc and doc["typrubrik"] != '':
        title = etree.SubElement(bibl_elem, "title")
        title.text = doc["typrubrik"]
    else:
        title = etree.SubElement(bibl_elem, "title")
        title.text = doc["titel"]
    if doc["dok_id"] is not None and doc["dok_id"] != '':
        idno = etree.SubElement(bibl_elem, "idno")
        idno.text = doc["dok_id"]
        idno.attrib["type"] = "rdwebb"
    if doc["beteckning"] is not None and doc["beteckning"] != '':
        nrrs = etree.SubElement(bibl_elem, "rs")
        nrrs.attrib["type"] = "number"
        nrrs.text = doc["beteckning"]
    if doc["organ"] is not None and doc["organ"] != '':
        orgname = etree.SubElement(bibl_elem, "orgName")
        orgname.attrib["type"] = "comitteeAbbreviation"
        orgname.text = doc["organ"]
    #refs
    refs = ["dokument_url_text", "dokument_url_html", "dokumentstatus_url_xml"]
    for ref in refs:
        if doc[ref] is not None and doc[ref] != '':
            _ = etree.SubElement(bibl_elem, "ref")
            _.attrib["type"] = ref
            _.text = doc[ref]
    if attachments is not None:
        if type(attachments['bilaga']) == dict:
            bilagor = [attachments['bilaga']]
        else:
            bilagor = attachments['bilaga']
        if len(bilagor) > 0:
            for bilaga in bilagor:
                if bilaga["fil_url"] is not None and bilaga["fil_url"] != '':
                    _ = etree.SubElement(bibl_elem, "ref")
                    typ = bilaga["filtyp"]
                    if typ is not None and typ != '':
                        _.attrib["type"] = typ
                    else:
                        _.attrib["type"] = "Misc."
                    _.text = bilaga["fil_url"]
    return root


def set_title_stmt(root, ns, titlestmt, author):
    stmtelem = root.find(f".//{ns}titleStmt")
    title_elem = etree.SubElement(stmtelem, "title")
    title_elem.text = titlestmt
    if author is not None and author != '':
        author_elem = etree.SubElement(stmtelem, "author")
        author_elem.text = author
    return root


