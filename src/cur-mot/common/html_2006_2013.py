from common.html_common import (
    add_to_docD,
    doc_D,
    extract_table,
    fail_miserably,
    handle_h1,
    handle_h2,
    handle_h3,
    handle_h4,
    handle_h5,
    handle_h6,
    handle_list,
    handle_table,
)
from common.xml_utils import dict_to_tei
from lxml import etree
from bs4 import BeautifulSoup as bs
from bs4 import NavigableString
from bs4 import UnicodeDammit as damnit
import inspect, json, re, sys




def get_depth(elem):
    if hasattr(elem, "contents") and elem.contents:
        return max([get_depth(c) for (c) in elem.contents]) + 1
    else:
        return 0

def parse_html_2006_2013(html, filename):

    def _ck_notes(_, look_for_notes):
        notes = []
        if _.descendants is not None:
            for d in _.descendants:
                if type(d) != NavigableString and \
                    d.attrs is not None and \
                    "href" in d.attrs and \
                    d.attrs["href"].startswith("#_ftn"):

                    #print("FOOTNOTE found", d.attrs["href"])

                    note = (d.get_text(" ").strip(), d.attrs["href"])
                        #print(_)
                        #sys.exit()
                    note_refs.append(note[1])
                    look_for_notes = True
                    notes.append(note)
                    #print(_, notes, look_for_notes)
        t =_.get_text(" ").strip()
        if len(notes) !=0 :
            for note in notes:
                #print(note)
                t = t.replace(note[0], f"<ref target=\"{note[1]}\">{note[0]}</ref>")
                #print(t)
        return t, look_for_notes


    def _handle_div(div, docD, current_div, div_d, p, dd, look_for_notes):
        dd += 1
        for e in div:
            docD, div_d, current_div, p, dd, look_for_notes = _sort_elems(e, docD, div_d, current_div, p, dd, look_for_notes)
        return docD, current_div, div_d, p, dd, look_for_notes


    def _handle_p(_, docD, div_d, current_div, look_for_notes):
        #print("   ~~handling P")
        if _.attrs is not None and "class" in _.attrs and "ft0" in _.attrs["class"]:
            docD, current_div, div_d = handle_h1(_, docD, div_d, current_div)
        elif _.attrs is not None and "style" in _.attrs and "font-weight: bold;" in _.attrs["style"]:
            docD, current_div, div_d = handle_h1(_, docD, div_d, current_div)
        else:
            if "pars" not in div_d:
                div_d['pars'] = []
            if _.get_text(' ').strip() is not None and _.get_text(' ').strip() != '':
                t, look_for_notes = _ck_notes(_, look_for_notes)
                div_d['pars'].append({"p": t})

        return docD, current_div, div_d, look_for_notes


    def _sort_elems(_, docD, div_d, current_div, p, dd, look_for_notes):
        #print("  ••", _.name, dd)
        if _.name == "div":
            docD, current_div, div_d, p, dd, look_for_notes = _handle_div(_, docD, current_div, div_d, p, dd, look_for_notes)
            #print("  ..DIV", dd)
        elif _.name == "p":
        #    print("      __-->", _.name)
            docD, current_div, div_d, look_for_notes = _handle_p(_, docD, div_d, current_div, look_for_notes)
        elif _.name == "h1":
            docD, current_div, div_d = handle_h1(_, docD, div_d, current_div)
            #p = True
        elif _.name == "h2":
            docD, current_div, div_d = handle_h2(_, docD, div_d, current_div)
            #p = True
        elif _.name == "h3":
            docD, current_div, div_d = handle_h3(_, docD, div_d, current_div)
            #p = True
        elif _.name == "h4":
            docD, current_div, div_d = handle_h4(_, docD, div_d, current_div)
            #p = True
        elif _.name == "h5":
            docD, current_div, div_d = handle_h5(_, docD, div_d, current_div)
            #p = True
        elif _.name == "h6":
            docD, current_div, div_d = handle_h6(_, docD, div_d, current_div)
            #p = True
        elif _.name == "hr":
            #print(_)
            pass
        elif _.name == "table":
            docD, current_div, div_d = handle_table(_, docD, div_d, current_div)
            #p = True
        elif _.name in ["ol", "ul"]:
            if "pars" not in div_d:
                div_d['pars'] = []
            div_d["pars"].append({"p": handle_list(_, _name=_.name)})
            #p = True
        elif _.name == "style":
            print("ignore style elem")
            #pass
        elif type(_) == NavigableString:
            pass
        else:
            print(_.name)

        return docD, div_d, current_div, p, dd, look_for_notes



    current_div = None
    div_d = {}
    p = False
    look_for_notes = False
    dd = 0
    docD = doc_D()
    soup = bs(f"<html><body>{html}</body></html>", 'html.parser')

    try:
        assert soup.html.body != None, f"soup.html.body is None: {filename}"
        #[print(_.name) for _ in soup]
    except:
        f"soup.html.body is None: {filename}"
        sys.exit()

    try:
        docD["header_title"] = soup.find("span", class_="sidhuvud_publikation").string.strip() + " " + \
                            soup.find("span", class_="sidhuvud_beteckning").string.strip() + " " + \
                            soup.find("span", class_="MotionarLista").string.strip() + " " + \
                            soup.find("h1", recursive=False).string.strip()
    except:
        pass
    body = soup.html.body

    if body.find("html") is not None:
        body = body.html.body
    for i, _ in enumerate(body):
        #print(f"{i}---")
        if type(_) == NavigableString:
            print("nav string")
            pass
        elif _.name is None:
            print("no-name elem")
            print(_)
        else:
            print(" •", _.name, current_div)
            docD, div_d, current_div, p, dd, look_for_notes = _sort_elems(_, docD, div_d, current_div, p, dd, look_for_notes)
            #print(" ~~", current_div)
        if p:
            print(_.name, docD)
        #print(f"---{i} {len(body)} \n")
        #print(docD)
    docD = add_to_docD(docD, div_d, current_div)

    return docD
