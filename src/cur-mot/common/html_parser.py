from common.html_2006_2013 import parse_html_2006_2013
from common.html_2014_2025 import parse_html_2014_2025
from common.xml_utils import dict_to_tei
from lxml import etree
from bs4 import BeautifulSoup as bs
from bs4 import NavigableString
from bs4 import UnicodeDammit as damnit
import inspect, json, re, sys




def populate_from_html(root, ns, html, py, filename):
    html_fns = {
        "2022-2025": parse_html_2014_2025,
        "2018-2021": parse_html_2014_2025,
        "2014-2017": parse_html_2014_2025,
        "2010-2013": parse_html_2006_2013,
        "2006-2009": parse_html_2006_2013,

    }
    # Here is some logic to decide which type of html we're dealing with
    #    and send to the relevant function.
    # Strategy: parse html to dict and then one fn to populate the tei body elems
    if int(py[:4]) >= 2022 and int(py[4:]) <= 2025:
        d = html_fns["2022-2025"](html, filename)
    elif int(py[:4]) >= 2018 and int(py[4:]) <= 2021:
        d = html_fns["2018-2021"](html, filename)
    elif int(py[:4]) >= 2014 and int(py[4:]) <= 2017:
        d = html_fns["2014-2017"](html, filename)
    elif int(py[:4]) >= 2010 and int(py[4:]) <= 2013:
        d = html_fns["2010-2013"](html, filename)
    elif int(py[:4]) >= 2006 and int(py[4:]) <= 2009:
        d = html_fns["2006-2009"](html, filename)
    else:
        print("\n\n\nHTML classifying strategy fails. whaaa....\n\n\n")
        sys.exit()


    if d is not None:
        root = dict_to_tei(d, root, ns)
    #else:
        """
        try:
            print("<p>"+html.replace("<br />", "\n")+"</p>")
            body = root.find(f".//{ns['tei_ns']}div[@type='motBody']")
            p = etree.fromstring("<p>"+html.replace("<br />", "\n")+"</p>")
            print(body, p)
            body.append(p)
        except:
            pass
        """
    return root
