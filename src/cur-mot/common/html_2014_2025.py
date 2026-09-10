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







def ck_body(D, n, _print, current_div):
    if _print:
        if "body" in D:
            #"""
            if type(D["body"]) == dict:
                print("~~~>", n, type(D["body"]), 0, current_div)
            """
            if D["body"] is None:
                print("~~~>", n, type(D["body"]), 0, current_div)
            else:
                print("~~~>", n, type(D["body"]), len(D["body"]), current_div)
        else:
            print("++++>", n, "no body", current_div)
            #"""





def parse_html_2014_2025(html, filename):

    # PRIVATE FNS

    def _ck_notes(_, look_for_notes):
        notes = []
        if _.descendants is not None:
            for d in _.descendants:
                if type(d) != NavigableString and \
                    d.attrs is not None and \
                    "href" in d.attrs and \
                    d.attrs["href"].startswith("#_ftn"):

                    #print("FOOTNOTE found", d.attrs["href"])

                    note = (d.get_text().strip(), d.attrs["href"])
                        #print(_)
                        #sys.exit()
                    note_refs.append(note[1])
                    look_for_notes = True
                    notes.append(note)
                    #print(_, notes, look_for_notes)
        t =_.get_text().strip()
        if len(notes) !=0 :
            for note in notes:
                #print(note)
                t = t.replace(note[0], f"<ref target=\"{note[1]}\">{note[0]}</ref>")
                #print(t)
        return t, look_for_notes




    #######################
    # DIV DIV DIV DIV DIV #
    #######################
    def _handle_div(_, docD, div_d, current_div):
        if "CC_Boilerplate_4" in _.attrs["style"]:
            docD = add_to_docD(docD, div_d, current_div)
            current_div = "förslag"
            if docD[current_div] is None:
                div_d = {}
            else:
                div_d = docD[current_div]
            #print(_.get_text().strip())
            div_d["heading"] = {"type":"h1", "text": _.get_text().strip()}

        elif "CC_Motivering_Rubrik" in _.attrs["style"]:
            docD = add_to_docD(docD, div_d, current_div)
            current_div = "body"
            #print(_.get_text().strip())
            div_d = {"heading":{"type":"h1", "text": _.get_text().strip()}, "pars": []}

        elif "CC_Underskrifter" in _.attrs["style"]:
            docD = add_to_docD(docD, div_d, current_div)
            current_div = "signature-block"
            div_d = {}

        elif ('Yrkande' in _.attrs["style"] or 'yrkande' in _.attrs["style"]) and current_div == "förslag":
            #print("THERE", _.attrs["style"])
            if "pars" not in div_d:
                div_d["pars"] = []
            if 'style' in _.attrs:
                m = yrkande_pat.search(_.attrs["style"])
                if m is not None:
                    p = m[1]
                else:
                    p = "p"
            div_d["pars"].append({p:_.get_text().strip()})

        elif "style" in _.attrs and \
            _.attrs["style"] == "-aw-sdt-tag:''" and \
            "Innehåll" in _.get_text():
            docD = add_to_docD(docD, div_d, current_div)
            current_div = "toc"
            div_d = {"pars": []}
            heading = None
            try:
                heading = _.find("p", "Rubrik1numrerat").get_text().strip()
                div_d["heading"] = {"text": heading, "type":"h1"}
            except:
                heading = None
            if not heading:
                try:
                    heading = _.find("h1").get_text().strip()
                    div_d["heading"] = {"text": heading, "type":"h1"}
                except:
                    heading = None
            if not heading:
                try:
                    heading = _.find("p", "TOCHeading").get_text().strip()
                    div_d["heading"] = {"text": heading, "type":"h1"}
                except:
                    heading = None
            if heading is None:
                print("heading is none.", _)
                sys.exit()
            for elem in _.find_all("p", {'class': ["TOC1", "TOC2", "TOC3", "TOC4"]}):
                #print(elem.attrs["class"])
                div_d["pars"].append({elem.attrs["class"][0]: elem.get_text().strip()})

        elif current_div == 'body' and _.table is not None:
            div_d['pars'].append({"table": extract_table(_.table, div_d)})

        elif ((current_div == 'body' or current_div == 'förslag') and \
            _.attrs is not None and \
            'style' in _.attrs and \
            'border-bottom' in _.attrs['style']) or \
            (current_div == 'epilogue'):
            for c in _.children:
                if c.name == 'h2':
                    if current_div =='body':
                        if docD["body"] is None:
                            docD["body"] = []
                        #print(current_div, docD)
                        docD[current_div].append(div_d)
                        ck_body(docD, 7, body_ck, current_div)
                        div_d = {"heading":
                                    {"type": "h2",
                                    "text": c.get_text().strip()},
                                    "pars": []}
                    else:
                        docD[current_div] = {"pars": [{"p": _.get_text().strip()}]}
                elif c.name == 'p':
                    div_d['pars'].append({'p': c.get_text().strip()})
                elif type(c) == NavigableString and \
                    c.name == None and \
                    c.get_text().strip() == '':
                    pass
                elif c.name in ["ul", "ol"]:
                    if "pars" not in div_d:
                        div_d['pars'] = []
                    div_d["pars"].append({"p": handle_list(_, _name=c.name)})
                    #if current_div == "body":
                    #    ck_body(docD, 14, body_ck, current_div)
                    #else:
                    current_div = "body"
                elif c.name == "h1":
                    docD, current_div, div_d = handle_h1(c, docD, div_d, current_div)
                elif c.name == "h2":
                    docD, current_div, div_d = _handle_h2(c, docD, div_d, current_div)
                else:
                    if type(c) == NavigableString:
                        print("child is navigable string", c, c.name, c.get_text())
                    elif c.attrs is not None:
                        print("unknown elemeent tyoe in div", c, c.attrs)
                    else:
                        print("unknown elemeent tyoe in div", c)
                    sys.exit()
        else:
            #print(_)
            x = inspect.getframeinfo(inspect.currentframe()).lineno
            print(docD)
            fail_miserably(f"unknown div type -- {_.name} {_.attrs} {current_div}\n warning from line {x}")

        return docD, current_div, div_d

    #############
    # P P P P P #
    #############
    def _handle_p(_, docD, div_d, current_div, look_for_notes):
    #    print(current_div, _.get_text().strip()[:25])
        if _.get_text().strip().endswith("nnehållsförteckning") or \
            ("class" in _.attrs and "TOCHeading" in _.attrs["class"]):
    #        print("~p", 1)
            docD = add_to_docD(docD, div_d, current_div)
            current_div = "toc"
            div_d = {"heading": {"text":_.get_text().strip(), "type":"h1"}, "pars": []}

        elif current_div == "toc" and "class" in _.attrs and \
            any([(x in ["TOC1", "TOC2", "TOC3", "TOC4"]) for x in _.attrs["class"]]):
    #        print("~p", 3)
            div_d["pars"].append({_.attrs["class"][0]: _.get_text().strip()})

        elif _.get_text().strip() == "Motivering":
    #        print("~p", 2, _.attrs)
            docD = add_to_docD(docD, div_d, current_div)
            current_div = "body"
            div_d = {"heading": {"text":_.get_text().strip(), "type":"h1"}, "pars":[]}

        elif "class" in _.attrs and "RubrikFrslagTIllRiksdagsbeslut" in _.attrs["class"]:
        #    print("~p", 4)
            docD = add_to_docD(docD, div_d, current_div)
            current_div = "förslag"
            if docD[current_div] is None:
                div_d = {}
            else:
                div_d = docD[current_div]
            #print(_.get_text().strip())
            div_d["heading"] = {"type":"h1", "text":_.get_text().strip()}
            ck_body(docD, 10, body_ck, current_div)

        elif _.get_text().strip().endswith("ammanfattning") and current_div != 'toc':
        #    print("~p", 5)
            docD = add_to_docD(docD, div_d, current_div)
            current_div = "summary"
            div_d = {"heading": {"type":"h1", "text": _.get_text().strip()}, "pars":[]}

        elif "Förslag till riksdagsbeslut" in _.get_text().strip() and \
            ("class" not in _.attrs or not any(["TOC" in x for x in _.attrs["class"]])):
        #    print("~p", 6)
            if docD['förslag'] is None:
                docD = add_to_docD(docD, div_d, current_div)
                current_div = "förslag"
                div_d = {"heading":{"type":"h1", "text": _.get_text().strip()}, "pars":[]}

            elif _.get_text().strip() is not None and _.get_text().strip() != '':
                t, look_for_notes = _ck_notes(_, look_for_notes)
                div_d['pars'].append({"p": t})

        elif current_div == 'toc' and \
            "class" in _.attrs and \
            _.attrs["class"][0] in ["TOC1", "TOC2", "TOC3", "TOC4"]:
        #    print("~p", 7)
            if "pars" not in div_d:
                div_d["pars"] = []
            div_d["pars"].append({_.attrs["class"][0]: _.get_text().strip()})

        elif _.attrs is not None and "class" in _.attrs and \
            (
                "Rubrik1numrerat" in _.attrs["class"] or \
                "Rubrik2numrerat" in _.attrs["class"] or \
                "Rubrik3numrerat" in _.attrs["class"]
            ):
        #    print("~p", 8)
            docD = add_to_docD(docD, div_d, current_div)
            #print("¡¡¡¡¡¡!!!!!!!!!!", _.attrs["class"][0][6])
            N = _.attrs["class"][0][6]
            current_div = "body"
            div_d = {"heading": {"type": f"h{N}", "text": _.get_text().strip()}, "pars": []}

        elif 'style' in _.attrs and "-aw-list-level-number:0;" in _.attrs['style']:
        #    print("~p", 9)
            if current_div == "förslag" and len(div_d["pars"]) > 0 and \
                not ('Yrkande' in _.attrs["style"] or 'yrkande' in _.attrs["style"]):
                docD[current_div] = div_d
                docD["body"] = []
            else:
                docD = add_to_docD(docD, div_d, current_div)
            current_div = "body"
            div_d = {"heading": {"type": "h1", "text": _.get_text().strip()}, "pars": []}

        elif _.attrs is not None and "class" in _.attrs and \
            (
                "Rubrik1numrerat" in _.attrs["class"] or \
                "Normalutanindragellerluft" in _.attrs["class"]
            ) and (
                current_div == 'body' or \
                current_div == 'summary' or \
                current_div == "toc" or \
                current_div == "förslag"
            ):
        #    print("~p", 10)
            if len(_.get_text().strip()) < 100:
                docD = add_to_docD(docD, div_d, current_div)
                current_div = "body"
                div_d = {"heading": {"type": "h1", "text": _.get_text().strip()}, "pars": []}
            else:
                header = False
                for d in _.descendants:
                    if type(d) is not NavigableString:
                        if d.attrs is not None and 'name' in d.attrs and \
                            (d.attrs['name'].startswith("_Toc") or d.attrs['name'] == "MotionStart"):
                            header = True
                if header:
                    if ("class" in _.attrs and "Klla" in _.attrs["class"]) or len(_.get_text().strip()) > 100:
                        t, look_for_notes = _ck_notes(_, look_for_notes)
                        div_d['pars'].append({'p': t})
                    else:
                        docD = add_to_docD(docD, div_d, current_div)
                        current_div = "body"
                        div_d = {"heading":{"type":"h1", "text":_.get_text().strip()}, "pars": []}

                else:
                    if "pars" not in div_d:
                        div_d['pars'] = []

                    if _.get_text().strip() is not None and _.get_text().strip() != '':
                        t, look_for_notes = _ck_notes(_, look_for_notes)
                        div_d['pars'].append({"p": t})


        elif "class" in _.attrs and \
            (
                "Rubrik1numrerat" in _.attrs["class"] or \
                "Normalutanindragellerluft" in _.attrs["class"]
            ) and "Förslag till riksdagsbeslut" in _.get_text():
        #    print("~p", 11)
            if current_div is not None:
                docD[current_div] = div_d
                div_d = {}
            current_div = "förslag"
            div_d["heading"] = {"type": "h1", "text": _.get_text().strip()}

        elif "style" in _.attrs and "CC_Underskrifter" in _.attrs["style"]:
        #    print("~p", 12)
            docD = add_to_docD(docD, div_d, current_div)
            current_div = "signature-block"
            div_d = {"heading": {"type": "h1", "text": _.get_text().strip()}, "pars": []}

        elif "class" in _.attrs and "Underskrifter" in _.attrs["class"]:
            print("~p 12-2", current_div)
            if current_div == 'signature-block':
                print("~p 12-2.1")
                div_d["pars"].append(_.get_text().strip())
            else:
                print("~p 12-2.2")
                docD = add_to_docD(docD, div_d, current_div)
                current_div = 'signature-block'
                div_d = {"heading": {'type':'h1', 'text': ''}, "pars": [_.get_text().strip()]}

        elif (
                'style' in _.attrs and \
                (
                    "-aw-list-level-number:1;" in _.attrs['style'] or \
                    "-aw-list-level-number:3;" in _.attrs['style']
                )
            ) or (
                _.attrs is not None and \
                "class" in _.attrs and \
                "Rubrik2numrerat" in _.attrs["class"] and \
                current_div == 'body'
            ):
        #    print("~p", 13)
            docD = add_to_docD(docD, div_d, current_div)
            current_div = "body"
            if "-aw-list-level-number:1;" in _.attrs['style']:
                div_d = {"heading": {"type": "h2", "text": _.get_text().strip()}, "pars": []}
            else:
                div_d = {"heading": {"type": "h3", "text": _.get_text().strip()}, "pars": []}

        elif len(_.find_all('span')) == 1 and _.find('span').attrs is not None and "style" in _.find('span').attrs and "font-weight:bold" in _.find('span').attrs["style"]:
        #    print("~p", 14)
            docD = add_to_docD(docD, div_d, current_div)
            current_div = "body"
            div_d = {"heading": {"type": "h1", "text": _.find('span').get_text().strip()}, "pars": []}

        elif current_div == "förslag":
        #    print("~p", 15)
            if _.find('a', {'name': "MotionsStart"}) is not None:
                docD[current_div] = div_d
                div_d = {"heading": {"type":"h1","text":None}, "pars": []}
                current_div = "body"
            elif "style" in _.attrs and 'Yrkande' in _.attrs["style"]:
                if "pars" not in div_d:
                    div_d["pars"] = []
                if 'style' in _.attrs:
                    m = yrkande_pat.search(_.attrs["style"])
                    if m is not None:
                        p = m[1]
                    else:
                        p = "p"
                    div_d["pars"].append({p:_.get_text().strip()})

        else:
        #    print("~p", 16)
            header = False
            for d in _.descendants:
                if type(d) is not NavigableString:
                    if d.attrs is not None and 'name' in d.attrs and \
                        (d.attrs['name'].startswith("_Toc") or d.attrs['name'] == "MotionStart"):
                        header = True
            if header:
                if ("class" in _.attrs and "Klla" in _.attrs["class"]) or len(_.get_text().strip()) > 100:
                    t, look_for_notes = _ck_notes(_, look_for_notes)
                    div_d['pars'].append({'p': t})
                else:
                    docD = add_to_docD(docD, div_d, current_div)
                    current_div = "body"
                    div_d = {"heading":{"type":"h1", "text":_.get_text().strip()}, "pars": []}


            else:
                if "pars" not in div_d:
                    div_d['pars'] = []
                if _.get_text().strip() is not None and _.get_text().strip() != '':
                    t, look_for_notes = _ck_notes(_, look_for_notes)
                    div_d['pars'].append({"p": t})

        return docD, current_div, div_d, look_for_notes

    ###########################
    # OL UL OL UL OL UL OL UL #
    ###########################
    def  _handle_ol_ul(_, docD, div_d, current_div):
    #    print("HERE!!!", current_div, div_d)
        if "pars" not in div_d:
            div_d['pars'] = []
        div_d["pars"].append({"p": handle_list(_)})
        return docD, current_div, div_d




    # PUBLIC


    docD = doc_D()
    body_ck = True
    soup = bs(html, "html.parser")
    yrkande_pat = re.compile(r"-aw-sdt-title:'(Yrkande\s+[0-9]+)\'")

    try:
        soup = soup.find("div", class_="pconf")
        assert soup is not None
    except:
        try:
            soup = bs(f"<html>{html}</html>", "html.parser")
        except:
            fail_miserably("  No soup 1 -- parse_html -- I don't know what to do!")
            with open("riksdagen-motions/no-soup.txt", "a+") as nosoup:
                nosoup.write(f"{filename}\n")
            return None
        t = []
        for _ in soup.descendants:
            if type(_) == NavigableString:
                t.append(_.strip())
            else:
                if _.string is not None:
                    t.append(_.string.strip())
        s = ' '.join([_.strip() for _ in t])
        #print(s)
        if "Motionen utgår" in s:
            docD['body'] = [{"heading": {"type":"h1", "text": "Utgår"}, "pars": [{"p": s}]}]
        elif "[~ konvertering pågår ~]" in s:
            docD['body'] = [{"heading": {"type":"h1", "text": "Utgår"}, "pars": [{"p": s}]}]
        else:
            fail_miserably("  No soup 2 -- parse_html -- I don't know what to do!")
            with open("riksdagen-motions/no-soup.txt", "a+") as nosoup:
                nosoup.write(f"{filename}\n")
            return None
    else:
        docD["header_title"] = soup.find("span", class_="sidhuvud_publikation").string.strip() + " " + \
                            soup.find("span", class_="sidhuvud_beteckning").string.strip() + " " + \
                            soup.find("span", class_="MotionarLista").string.strip() + " " + \
                            soup.find("h1", recursive=False).string.strip()
        section_1 = soup.find("div", class_="Section1")
        current_div = None
        look_for_notes = False
        note_refs = []
        div_d = {}
        for i_, _ in enumerate(section_1):

            #print(current_div)
            _print = False
            if type(_) == NavigableString:
                if _.string.strip() is not None and _.string.strip() != '':
                    pass
                    #print(" x", _, f"|{_.string.strip()}|")
            else:
            #    print(_.name, current_div)
                #if i_ < 75:
                #    print("-~-~>", current_div, docD)
    # DIV   #   #
                if _.name == "div":
                    docD, current_div, div_d = _handle_div(_, docD, div_d, current_div)
    # HR    #   #
                elif _.name == "hr":
                    print("FOUND HR")
                    if "style" in _.attr and "-aw-footnote" in _.attrs["style"]:
                        print("footnotes")
                        sys.exit()
    # H1    #   #
                elif _.name == "h1":
                    docD, current_div, div_d = handle_h1(_, docD, div_d, current_div)
    # H2    #   #
                elif _.name == "h2":
                    docD, current_div, div_d = handle_h2(_, docD, div_d, current_div)
    # H3    #   #
                elif _.name == "h3":
                    docD, current_div, div_d = handle_h3(_, docD, div_d, current_div)
    # H4    #   #
                elif _.name == "h4":
                    docD, current_div, div_d = _handle_h4(_, docD, div_d, current_div)
    # H5    #   #
                elif _.name == "h5":
                    docD, current_div, div_d = _handle_h5(_, docD, div_d, current_div)
    # H6    #   #
                elif _.name == "h6":
                    docD, current_div, div_d = _handle_h6(_, docD, div_d, current_div)
    # P     #   #
                elif _.name == "p":
                    docD, current_div, div_d, look_for_notes = _handle_p(_, docD, div_d, current_div, look_for_notes)
    # lists #   #
                elif _.name == "ol" or _.name == "ul":
                    docD, current_div, div_d = _handle_ol_ul(_, docD, div_d, current_div)
    # table #   #
                elif _.name == "table":
                    docD, current_div, div_d = handle_table(_, docD, div_d, current_div)
                elif _.name in ["br", "img", "span"]:
                    pass
                else:
                    print("unknown tag:", _.name)
                    sys.exit()

        docD = add_to_docD(docD, div_d, current_div)

        if look_for_notes:
            docD["notes"] = []
            for n in note_refs:
            #    print('~~~', n)
                N = soup.find("div", {"id":n[1:]})
            #    print(N.get_text().strip())
                docD["notes"].append((n, N.get_text().strip()))

    return docD
