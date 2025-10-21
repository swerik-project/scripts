from common.cleanup_xml import clean_xml
from lxml import etree
from pyparlaclarin.refine import format_paragraph
from pyriksdagen.utils import get_formatted_uuid
import json, re, sys


def dict_to_tei(d, root, ns):
    ### BEGIN DEBUG ###
    #print(d)
    #print(d['notes'])
    #if d['notes'] is not None:
    #print(json.dumps(d['body'], indent=2, ensure_ascii=False))
    with open("riksdagen-motions/tmp.json", "w+") as out:
        json.dump(d, out, ensure_ascii=False, indent=4)

    ### END DEBUG ###
    def _concat_seedtext(seedtext, addition):
        a = addition[:]
        seedtext += "\n" + a
        #print("concat:", len(seedtext), len(addition[:100]), len(seedtext)+len(a), addition[:100])
        return seedtext

    def _handle_list(elem, _list, seedtext):
        for item in _list:
            item = tuple(item.items())[0]
            #li = etree.SubElement(elem, 'item')
            li = etree.fromstring(f"<item>{item[1]}</item>".replace("&", "&amp;"))
            seedtext = _concat_seedtext(seedtext, item[1])
            li.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
            elem.append(li)
        return _list, seedtext

    def _handle_table(table, par, seedtext):
        if "header" in par[1]:
            row = etree.SubElement(table, "row")
            row.attrib["style"] = "label"
            for val in par[1]["header"][0]:
                cell = etree.SubElement(row, "cell")
                cell.text = val
                seedtext = _concat_seedtext(seedtext, val)
            row.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
        for r in par[1]["rows"]:
            row = etree.SubElement(table, "row")
            seedtext = _concat_seedtext(seedtext, "row")
            for val in r:
                cell = etree.SubElement(row, "cell")
                cell.text = val
                seedtext = _concat_seedtext(seedtext, val)
            row.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
        return table, seedtext

    seedtext = root.attrib[f"{ns['xml_ns']}id"]
    title = root.find(f".//{ns['tei_ns']}div[@type='motTitle']")
    title.text = d["header_title"]
    body = root.find(f".//{ns['tei_ns']}div[@type='motBody']")
    if d["summary"] is not None:
        sumdiv = etree.SubElement(body, 'div')
        sumdiv.attrib["type"] = "motSummary"
        sumhead = etree.SubElement(sumdiv, "head")
        sht = d["summary"]["heading"]["text"]
        sumhead.text = sht
        #toclist = etree.SubElement(tocdiv, "list")
        seetext = _concat_seedtext(seedtext, sht)
        sumhead.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)

        for par in d["summary"]["pars"]:
            par = tuple(par.items())[0]
            #print("~~summary par:", par)
            if type(par[1]) == str:
                parelem = etree.fromstring(f"<p>{par[1]}</p>".replace("&", "&amp;"))
                seedtext = _concat_seedtext(seedtext, par[1])
                parelem.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
                sumdiv.append(parelem)
            elif type(par[1]) == dict:
                list_tup = tuple(par[1].items())[0]
                if list_tup[0] in ["ul", "ol"]:
                    sumlist = etree.SubElement(sumdiv, 'list')
                    sumlist.attrib["style"] = list_tup[0]
                    sumlist, seedtext = _handle_list(sumlist, list_tup[1], seedtext)
                elif par[0] == 'table':
                    table = etree.SubElement(sumdiv, "table")
                    table, seedtext = _handle_table(table, par, seedtext)
                else:
                    print(json.dumps(d["summary"], indent=2, ensure_ascii=False))
                    print("unknown list type 66", list_tup[0])
                    sys.exit()
            else:
                print("   GAH!!!")
                sys.exit()

    if d["toc"] is not None:
        #print("TOC")
        tocdiv = etree.SubElement(body, 'div')
        tocdiv.attrib["type"] = "TOC"
        tochead = etree.SubElement(tocdiv, "head")
        #print(d['toc'])
        tochead.text = d["toc"]["heading"]["text"]
        seedtext = _concat_seedtext(seedtext, d["toc"]["heading"]["text"])
        tochead.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
        toclist = etree.SubElement(tocdiv, "list")
        for par in d["toc"]["pars"]:

            if type(par) == dict:
                if "table" in par:
                    for row in par["table"]["rows"]:
                        il = len(row[0].split('.'))
                        t = ' '.join(row)
                        item = etree.SubElement(toclist, "item")
                        item.attrib["style"] = f"indent-level: {il}"
                        item.text = t
                        seedtext = _concat_seedtext(seedtext, t)
                elif "p" in par:
                    item = etree.SubElement(toclist, "item")
                    item.text = par["p"]
                else:
                    print("unknown dict type in TOC")
                    print(par)
                    sys.exit()
            else:
                par = tuple(par.items())[0]
                item = etree.SubElement(toclist, "item")
                item.attrib["style"] = f"indent-level: {par[0][3:]}"
                item.text = par[1]
                seedtext = _concat_seedtext(seedtext, par[1])
        seedtext = _concat_seedtext(seedtext, d["toc"]["heading"]["text"])
        tocdiv.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
    if d["förslag"] is not None:
        #print("FÖRSLAG", d["förslag"])
        fdiv = etree.SubElement(body, 'div')
        fdiv.attrib["type"] = "motProposal"
        fdiv.attrib["subtype"] = "förslag"
        fhead = etree.SubElement(fdiv, "head")
        fhead.text = d["förslag"]["heading"]["text"]
        seedtext = _concat_seedtext(seedtext, d["förslag"]["heading"]["text"])
        fhead.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)

        if "pars" in d["förslag"]:
            if any([any([k.startswith("Yrkande") for k,v in _.items()]) for _ in d['förslag']['pars']]):
                for par in d["förslag"]["pars"]:
                    par = tuple(par.items())[0]
                    flist = etree.SubElement(fdiv, 'list')
                    if type(par[1]) == str:
                        item = etree.fromstring(f"<item>{par[1]}</item>".replace("&", "&amp;"))
                        if par[0].startswith("Yrkande"):
                            item.attrib["sortKey"] = par[0]
                        seedtext = _concat_seedtext(seedtext, par[1])
                        item.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
                        flist.append(item)
                    elif type(par[1]) == dict:
                        list_tup = tuple(par[1].items())[0]
                        if list_tup[0] in ["ul", "ol"]:
                            flist = etree.SubElement(fdiv, 'list')
                            flist.attrib["style"] = list_tup[0]
                            flist, seedtext = _handle_list(flist, list_tup[1], seedtext)
                        elif par[0] == "table":
                            table = etree.SubElement(fdiv, "table")
                            table, seedtext = _handle_table(table, par, seedtext)
                        else:
                            print(list_tup)
                            print("unknown list type 134", list_tup[0])
                            sys.exit()

                    else:
                        print(f"unsupported förslag paragraph format: {type(par[1])}\n{par[1]}")
                        sys.exit()
            else:
                for par in d["förslag"]["pars"]:
                    par = tuple(par.items())[0]
                    flist = etree.SubElement(fdiv, 'list')
                    #print("PAR", type(par[1]))
                    if type(par[1]) == str:
                    #    print('  if')
                        if par[0].startswith("Yrkande"):
                            #item = etree.SubElement(flist, "item")
                            item = etree.fromstring(f"<item>{par[1]}</itme>")
                            item.attrib["sortKey"] = par[0]
                            seedtext = _concat_seedtext(seedtext, par[1])
                            item.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
                            flist.append(item)
                        else:
                            #p = etree.SubElement(flist, "p")
                            #print(par[1])
                            p = etree.fromstring(f"<p>{par[1].replace('&', '&amp;')}</p>")
                            seedtext = _concat_seedtext(seedtext, par[1])
                            p.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
                            fdiv.append(p)
                   #     item = etree.SubElement(fdiv, "p")
                   #     item.text = par[1]
                   #     seedtext = _concat_seedtext(seedtext, par[1])
                   #    item.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
                    elif type(par[1]) == dict:
                    #    print('  elif')
                    #    print("-- dict")
                        list_tup = tuple(par[1].items())[0]
                        if list_tup[0] in ["ul", "ol"]:
                            list_elem = etree.SubElement(fdiv, "list")
                            list_elem.attrib["style"] = list_tup[0]
                            list_elem, seedtext = _handle_list(list_elem, list_tup[1], seedtext)
                        elif par[0] == 'table':
                            table = etree.SubElement(fdiv, "table")
                            table, seedtext = _handle_table(table, par, seedtext)
                        else:
                            print("unknown list type 143", list_tup)
                            sys.exit()

                    else:
                        print(f"unsupported förslag paragraph format: {type(par[1])}\n{par[1]}")
                        sys.exit()
                    #print(par[1])
    cdiv = etree.SubElement(body, 'div')
    cdiv.attrib["type"] = "motContent"
    for section in d["body"]:
        #print("BODY SECTION")
        if section["heading"]["text"] is not None:
            shead = etree.SubElement(cdiv, "head")
            shead.attrib["type"] = section["heading"]["type"]
            shead.text = section["heading"]["text"]
            seedtext = _concat_seedtext(seedtext, section["heading"]["text"])
            shead.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
        for par in section["pars"]:
            #print('~~', par)
            par = tuple(par.items())[0]

            if type(par[1]) == str:
                #print(f"<p>{par[1]}</p>"[:])
                T = par[1].replace('&', '&amp;')
                #print('a:', T)
                T = re.sub(r'<\s?(https?://(.*))>', r'\1', T)
                T = re.sub(r'<66', '&lt;66', T)
                #print('b:', T)
                #
                T = T.replace('<', '&lt;')
                T = T.replace('>', '&gt;')
                print(T)
                parelem = etree.fromstring(f"<p>{T}</p>")
                #print(parelem.text)
                seedtext = _concat_seedtext(seedtext, par[1])
                parelem.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
                #print(parelem, parelem.attrib)
                #print(etree.tostring(parelem))
                cdiv.append(parelem)
            elif type(par[1]) == dict:
                list_tup = tuple(par[1].items())[0]
                if list_tup[0] in ["ul", "ol"]:
                    list_elem = etree.SubElement(cdiv, "list")
                    list_elem.attrib["style"] = list_tup[0]
                    list_elem, seedtext = _handle_list(list_elem, list_tup[1], seedtext)
                elif par[0] == 'table':
                    table = etree.SubElement(cdiv, "table")
                    table, seedtext = _handle_table(table, par, seedtext)
                else:
                    print("unknown par dict type", par)
                    sys.exit()
            else:
                print(f"unsupported paragraph format: {type(par[1])}\n{par[1]}")
                sys.exit()

    if d["signature-block"] is not None:
        sigdiv = etree.SubElement(body, "div")
        sigdiv.attrib["type"] = "motSignatures"
        sigl = etree.SubElement(sigdiv, 'list')
        for par in d["signature-block"]["pars"]:
            if type(par) == str:
                if par.strip() is not None and par.strip() != '':
                    sigitem = etree.SubElement(sigl, "item")
                    sigitem.text = par
                    seedtext = _concat_seedtext(seedtext, par)
                    sigitem.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
            elif type(par) == dict:
                par = [tuple([k, v]) for k,v in par.items()][0]
                #print(par)
                if par[1].strip() is not None and par[1].strip() != '':
                    if par[1].strip().startswith("<ref target="):
                        m = re.search(r'(<ref\starget=\"(#_ftn\d{1,3})\">(\[\d{1,3}\])</ref>)(.*)', par[1].strip())
                        if m:
                            if d["notes"] is None:
                                d["notes"] = []
                            d["notes"].append([m[2], m[3] + ' ' + m[4].strip()])
                            sigitem = etree.Element("item")
                            sigitem.append(etree.fromstring(m[1]))
                            seedtext = _concat_seedtext(seedtext, m[1])
                            sigitem.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
                            sigl.append(sigitem)

                        else:
                            print("WARNING -- whasssaaaaat?!? 4", par)
                    else:
                        print("WARNING -- whasssaaaaat?!? 3", par)
                else:
                    print("WARNING -- whasssaaaaat?!? 2", par)
            else:
                print("WARNING -- whasssaaaaat?!? 1", par)
    if d["epilogue"] is not None:
        epidiv = etree.SubElement(body, "div")
        epidiv.attrib["type"] = "motEpilogue"

        if d["epilogue"]["heading"]["text"] is not None:
            epihead = etree.SubElement(epidiv, "head")
            epihead.attrib["type"] = d["epilogue"]["heading"]["type"]
            epihead.text = d["epilogue"]["heading"]["text"]
            seedtext = _concat_seedtext(seedtext, d["epilogue"]["heading"]["text"])
            epihead.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
        for par in d["epilogue"]["pars"]:
            if type(par) == str:
                if par.strip() is not None and par.strip() != '':
                    epip = etree.SubElement(epidiv, "p")
                    epip.text = par
                    seedtext = _concat_seedtext(seedtext, par)
                    epip.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
            elif type(par) == dict:
                par = [tuple([k, v]) for k,v in par.items()][0]
                #print(par)
                if par[0] == 'table':
                    table = etree.SubElement(epidiv, "table")
                    table, seedtext = _handle_table(table, par, seedtext)
                elif type(par[1]) == dict:
                    subpars = [tuple([k, v]) for k, v in par[1].items()]
                    for sp in subpars:
                        if sp[0] in ["ol", "ul"]:
                            epip = etree.SubElement(epidiv, "p")
                            _list = etree.SubElement(epip, "list")
                            _list, seedtext = _handle_list(_list, sp[1], seedtext)
                        else:
                            print("fail, line 257")
                            sys.exit()
                elif par[1].strip() is not None and par[1].strip() != '':
                    if par[1].strip().startswith("<ref target="):
                        m = re.search(r'(<ref\starget=\"(#_ftn\d{1,3})\">(\[\d{1,3}\])</ref>)(.{1,})', par[1].strip())
                        if m:
                            if d["notes"] is None:
                                d["notes"] = []
                            d["notes"].append([m[2], m[3] + ' ' + m[4].strip()])
                            epip = etree.Element(par[0])
                            epip.append(etree.fromstring(m[1]))
                            seedtext = _concat_seedtext(seedtext, m[1])
                            epip.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
                            epidiv.append(epip)

                        else:
                            print("WARNING -- whasssaaaaat?!? 4.1", par)
                    else:
                        epip = etree.Element(par[0])
                        epip.text = par[1]
                        seedtext = _concat_seedtext(seedtext, par[1])
                        epip.attrib[f"{ns['xml_ns']}id"] = get_formatted_uuid(seed=seedtext)
                        epidiv.append(epip)
                #else:
                #    print("WARNING -- whasssaaaaat?!? 2.1", par)
            else:
                print("WARNING -- whasssaaaaat?!? 1.1", par)
    if d["notes"] is not None:
        #print(json.dumps(d, indent=2))
        notes_D = {}
        noteGD = etree.SubElement(body, "div")
        noteGD.attrib["type"] = "motNotes"
        noteG = etree.SubElement(noteGD, "noteGrp")
        for n in d["notes"]:
        #    print("N:", n)
            note = etree.SubElement(noteG, "note")
            note.text = n[1]
            seedtext = _concat_seedtext(seedtext, n[1])
            note_id = get_formatted_uuid(seed=seedtext)
            note.attrib[f"{ns['xml_ns']}id"] = note_id
            notes_D[n[0]] = note_id
            #print('~\n', note_id, note, note.attrib, n)
        #print("NOTES:", notes_D)
        for k, v in notes_D.items():
            print(f"|{k}|{v}|")
            try:
                rev = root.find(f'.//ref[@target="{k}"]')
            except:
                rev = root.find(f'.//{ns["tei_ns"]}ref[@target="{k}"]')
            #print(rev)
            try:
                rev.attrib["target"] = v
            except:
                print("can't find reference ID")
                print(k, v)
                sys.exit()
    return root

def write_xml(root, name_, path_=None):
    def _indent_p(div, padding):
        #if div.text is not None:
        #    print(div.tag, div.text[:15])
        #else:
        #    print(div.tag)
        for p in div:
            #print(p.tag, len(p))
            #if p.text is not None:
            #    print(" ~", div.tag, p.text[:15])
            #else:
            #    print(" ~", div.tag)
            if p.text and len(p.text.strip()) > 0:
                if len(p) == 0:
                    p.text = format_paragraph(p.text, spaces=padding+2)
                else:
                    #print(p.text)
                    p.text = format_paragraph(p.text, spaces=padding+2) + '  '
                    for ei, e in enumerate(p):
                        if e.tail is not None and e.tail.strip() != '':
                            if ei == len(p)-1:
                                e.tail = format_paragraph(e.tail, spaces = padding+2)
                            else:
                                e.tail = format_paragraph(e.tail, spaces = padding+2) + '  '
                        elif e.tail is None and ei == len(p)-1:
                            e.tail = "\n" + ' '*padding

            elif not p.tag.endswith("pb"):
                p.text = "\n" + ' '*(int(padding)+2)
            for olx, ol in enumerate(p):
                if ol.tag == "ref":
                    ol.text = format_paragraph(ol.text.strip(), spaces=padding+4)
                    continue
                if ol.text:
                    #print("~~:", ol.tag, ol.text[:15])
                    ol.text = format_paragraph(ol.text.strip(), spaces = padding+4)
                #else:
                #    print("~~:", ol.tag, ol.text)
                if olx < len(p)-1:
                    ol.tail = "\n" + ' '*(int(padding)+2)
                else:
                    ol.tail = "\n" + ' '*int(padding)
                if 'type' in ol.attrib:
                    del ol.attrib['type']
                etree.indent(ol, '              ')
                for lix, li in enumerate(ol):

                    if li.text and len(li.text) > 0:
                        #print("~~~~", li.tag, li.text[:15])
                        li.text = f"\n{' '*(int(padding)+6)}{li.text.strip()}\n{' '*(int(padding)+4)}"
                    #else:
                    #    print("~~~~", li.tag, li.text)
                    if lix == len(ol)-1:
                        li.tail = "\n" + " "*(int(padding)+2)
                #if olx == len(p)-1:
                #    ol.tail = "\n" + " "*int(padding)
                #else:
                #    ol.tail = "\n" + " "*(int(padding)+2)
        return div

    def _format_paragraphs(root, padding=10, ns="{http://www.tei-c.org/ns/1.0}"):
        tei_ns = "{http://www.tei-c.org/ns/1.0}"
        xml_ns = "{http://www.w3.org/XML/1998/namespace}"
        header = root.find("text/body/div[@type='motHeader']")
        fws = root.findall(f".//{tei_ns}body/fw")
        fws.extend(root.findall(f".//{tei_ns}body/{tei_ns}fw"))
        for fw in fws:
            fw.text =  format_paragraph(fw.text, padding-2)
        if header is None:
            header = root.find(f"{tei_ns}text/{tei_ns}body/{tei_ns}div[@type='motHeader']")
        body = root.find("text/body/div[@type='motBody']")
        if body is None:
            body = root.find(f"{tei_ns}text/{tei_ns}body/{tei_ns}div[@type='motBody']")
        #print(body, len(body))
        if body is not None:
            for div in body:
                #print(type(body))
                #print("xx", div.tag)
                div = _indent_p(div, padding)
                #for div1 in div:
                #    div1 = _indent_p(div1, padding+2)
        if header is not None:
            for div in header:
                div = _indent_p(div, padding)
        return root

    root, write = clean_xml(root)
    if write:
        root = _format_paragraphs(root, padding=10)
        b = etree.tostring(
            root,
            pretty_print=True,
            encoding="utf-8",
            xml_declaration=True
        )
        if path_:
            if not os.path.exists(path_):
                os.mkdir(path_)
            out = f"{path_}/{name_}"
        else:
            out = name_
        with open(out, "wb") as f:
            f.write(b)


def parse_xml(_path, get_ns = True):
    """
    Parse a protocol, return root element (and namespace defnitions).

    Args:
        protocol_path (str): protocol path
        get_ns (bool): also return namespace dict

    Returns:
        tuple/etree._Element: root and an optional namespace dict
    """
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.parse(_path, parser).getroot()
    if get_ns:
        tei_ns = "{http://www.tei-c.org/ns/1.0}"
        xml_ns = "{http://www.w3.org/XML/1998/namespace}"
        return root, {"tei_ns":tei_ns, "xml_ns":xml_ns}
    else:
        return root
