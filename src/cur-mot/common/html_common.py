from bs4 import NavigableString




def add_to_docD(docD, div_d, current_div):
    #print("~~~> add to docD")
    if current_div == "body":
        if docD["body"] is None:
            docD["body"] = []
        docD["body"].append(div_d)
    elif current_div is not None:
        #print("current_div = ", current_div)
        #print("div_d = ", div_d)
        docD[current_div] = div_d
    return docD


def extract_table(_, div_d):
    hrow = []
    rows = []
    for tri, tr in enumerate(_.find_all("tr")):
        header = False
        if tri == 0:
            if any([(type(c) != NavigableString and \
                        c.attrs is not None and \
                        "style" in c.attrs and \
                        "border-bottom-style:solid" in c.attrs["style"]) for c in tr.children]):
    #            print("HEADER FOUND")
                header = True
        row = [c.get_text(" ").replace('\xa0',' ').strip() for c in tr.children \
                    if (c.get_text(" ").strip() is not None and\
                        c.get_text(" ").strip() != '')]
        #print(row)
        if header:
            hrow.append(row)
        else:
            rows.append(row)
    if "pars" not in div_d:
        div_d['pars'] = []
    table = {}
    if len(hrow) > 0:
        table["header"] = hrow
    table["rows"] = rows
    return table


def fail_miserably(m):
    #print("\n\n\n\t¡¡¡Failing Miserably!!!\n\t-----------------------\n\n\n\tbecause:\n")
    print(m)
    #print("\n\n\n")
    #sys.exit()


def doc_D():
    return {
            "header_title": None,            # str
            "header_submissionInfo": None,   #
            "summary": None,                 # dict
            "toc": None,                     # dict
            "förslag": None,
            "att-satser": None,
            "signature-block": None,
            "epilogue": None,
            "notes": None,
            "body": None,                    # list
        }





#####################
# H1 H1 H1 H1 H1 H1 #
#####################
def handle_h1(_, docD, div_d, current_div):
    if "Förslag till riksdagsbeslut" in _.get_text(" ") or ("class" in _.attrs and "Förslagsrubrik" in _.attrs["class"]):
        #print(_.text)
        docD = add_to_docD(docD, div_d, current_div)
        if docD["förslag"] is None:
        #    print("  --forslag")
            current_div = "förslag"
            if docD[current_div] is None:
                div_d = {}
            else:
                div_d = docD[current_div]
            div_d["heading"] = {"type":"h1","text": _.get_text(" ").strip()}
        else:
        #    print("  --body")
            current_div = "body"
            div_d = {"heading": {"type": "h1", "text": _.get_text(" ").strip()}, "pars": []}
    elif "Innehåll" in _.get_text(" "):
        docD = add_to_docD(docD, div_d, current_div)
        div_d = {"heading": {"text":_.get_text(" ").strip(), "type":"h1"}, "pars": []}
        current_div = "toc"

    elif _.get_text(" ").strip().endswith("ammanfattning"):
        if current_div is not None:
        #    print(current_div)
            docD = add_to_docD(docD, div_d, current_div)
        current_div = "summary"
        div_d = {"heading": {"type":"h1", "text": _.get_text(" ").strip()}, "pars": []}

    else:
        docD = add_to_docD(docD, div_d, current_div)
        current_div = "body"
        div_d = {"heading": {"type": "h1", "text": _.get_text(" ").strip()}, "pars": []}

    return docD, current_div, div_d


##################
# H2 H2 H2 H2 H2 #
##################
def handle_h2(_, docD, div_d, current_div):
    docD = add_to_docD(docD, div_d, current_div)
    current_div = "body"
    div_d = {"heading": {"type": "h2", "text": _.get_text(" ").strip()}, "pars": []}
    return docD, current_div, div_d


###############
# H3 H3 H3 H3 #
###############
def handle_h3(_, docD, div_d, current_div):
    docD = add_to_docD(docD, div_d, current_div)
    current_div = "body"
    div_d = {"heading": {"type": "h3", "text": _.get_text(" ").strip()}, "pars": []}
    return docD, current_div, div_d


############
# H4 H4 H4 #
############
def handle_h4(_, docD, div_d, current_div):
    mstart = False
    for ix, x in enumerate(_.descendants):
    #    print(ix)
        if type(x) != NavigableString:
    #        print(x.attrs)
            if x.attrs is not None:# and \
    #            print("  .")
                if "name" in x.attrs:# and \
    #                print("    .")
                    if x.attrs["name"] == "MotionsStart":
    #                    print("      .")
                        mstart = True
    if mstart == True:
        print(",,")
        if current_div == "body":
            if docD["body"] is None:
                docD["body"] = []
            docD["body"].append(div_d)
        elif current_div is not None:
            docD[current_div] = div_d
        current_div = "body"
        docD["body"] = []
        div_d = {"heading": {"text":_.get_text(" ").strip(), "type":"h1"}, "pars":[]}
    else:
        #print(current_div)
        if current_div == 'body':
            if docD['body'] is None:
                docD['body'] = []
            docD['body'].append(div_d)
            div_d = {"heading": {"type": "h4", "text": _.get_text(" ").strip()}, "pars": []}
        elif current_div == "toc":
            docD[current_div] = div_d
            div_d = {"heading": {"type": "h4", "text": _.get_text(" ").strip()}, "pars": []}
            current_div = "body"
        else:
            print("unknown type of h4")
            print(_)
            sys.exit()
    return docD, current_div, div_d


#########
# H5 H5 #
#########
def handle_h5(_, docD, div_d, current_div):
    docD = add_to_docD(docD, div_d, current_div)
    current_div = "body"
    div_d = {"heading": {"type": "h5", "text": _.get_text(" ").strip()}, "pars": []}
    return docD, current_div, div_d

######
# H6 #
######
def handle_h6(_, docD, div_d, current_div):
    docD = add_to_docD(docD, div_d, current_div)
    current_div = "body"
    div_d = {"heading": {"type": "h6", "text": _.get_text(" ").strip()}, "pars": []}
    return docD, current_div, div_d


#########
# LISTS #
# #########
def handle_list(list_elem, _name=None):
    if not _name:
        _name = list_elem.name
    _list = {_name: []}
    for li in list_elem:
        if li.get_text(" ").strip() is not None and li.get_text(" ").strip() != '':
            _list[_name].append({"li": li.get_text(" ").strip()})
    return _list


#################################
# TABLE TABLE TABLE TABLE TABLE #
#################################
def handle_table(_, docD, div_d, current_div):
    if current_div == "signature-block" or \
        len(_.find_all("p",'Underskrifter')) > 0:
        if current_div != 'signature-block':
            docD = add_to_docD(docD, div_d, current_div)
            current_div = 'signature-block'
            div_d = {}
        if "pars" not in div_d:
            div_d['pars'] = []
        people = _.find_all("p",'Underskrifter')
        for person in people:
            t = person.get_text(" ").strip()
            if t is not None and t != '':
                div_d['pars'].append(t)
        docD[current_div] = div_d
        current_div = "epilogue"
        div_d = {"heading":{"type":"h1", "text": None}, "pars": []}
    else:
        if 'pars' not in div_d:
            div_d['pars'] = []
        div_d['pars'].append({"table": extract_table(_, div_d)})
    return docD, current_div, div_d
