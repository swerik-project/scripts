from lxml import etree
from pyriksdagen.utils import (
    XML_NS,
    TEI_NS,
)

def del_empty_elems(root, ns=None):
    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #for e in root.iter():
    #    print(e.tag, e.attrib)
    #print("-----------------------------------------------")
    write = True
    for topdiv in root.findall(f".//{TEI_NS}text/{TEI_NS}body/{TEI_NS}div"):
        #print(topdiv.tag, topdiv.attrib, len(topdiv))
        deldiv = []
        subdivs = topdiv.findall(f"{TEI_NS}div")
        subdivs.extend(topdiv.findall("div"))
        if not subdivs:
            continue
        for dix, div in enumerate(subdivs):
        #    print(' ~ ', div.tag, div.attrib)
            delp = []
            for p in div:
        #        print('   p    ', p.tag, div.attrib)
                dellist = []
                if len(p) == 0 and (p.text is None or len(p.text.strip()) == 0):
                    delp.append(p)
                elif len(p) > 0:
                    for list_ in p:
                        if len(list_) == 0 and (list_.text is None or len(list.text.strip()) == 0):
                            dellist.append(list_)
                for _ in dellist:
        #            print("dellist", _, _.attrib)
                    if _.tag in ["pb", f"{TEI_NS}pb", f"{XML_NS}pb"] or \
                        (_.attrib is not None and \
                        'type' in _.attrib and \
                        _.attrib['type'] in ['motHeader', "motTitle", "motSubmissionInfo"]):
                        pass
                    else:
                        _.getparent().remove(_)
            #print(f"------------- RM p x {len(delp)} ---------")
            for _ in delp:
         #       print("DELP", _, _.attrib)
                if _.tag in ["pb", f"{TEI_NS}pb", f"{XML_NS}pb"] or \
                    (_.attrib is not None and \
                    'type' in _.attrib and \
                    _.attrib['type'] in ['motHeader', "motTitle", "motSubmissionInfo"]):
                    pass
                else:
                    _.getparent().remove(_)
            if len(div) == 0:
                deldiv.append(div)
        #print(f"~~~~~~~~~~~~~~~~~~~~~~~~ RM div x {len(deldiv)} ~~~~~~~~~~~~~~~~~~~~~")
        for _ in deldiv:
        #    print("deldiv", _, _.attrib)
            if _.tag in ["pb", f"{TEI_NS}pb", f"{XML_NS}pb"] or \
                (_.attrib is not None and \
                'type' in _.attrib and \
                _.attrib['type'] in ['motHeader', "motTitle", "motSubmissionInfo"]):
                pass
            else:
        #        print("div", _, _.attrib)
                _.getparent().remove(_)


        if topdiv.attrib["type"] == "motBody":
            if len(topdiv) == 0:
                write = False
                print("NO WRITE")
            #else:
            #    print("WRITE")
    dele = []
    for tag in root.find(f".//{TEI_NS}text/{TEI_NS}body").iter():

        #print(tag, tag.attrib)
        if tag.tag not in ["row", "cell", "pb", f"{TEI_NS}row", f"{TEI_NS}cell", f"{TEI_NS}pb"]:
            if len(tag) == 0 and (tag.text is None or tag.text.strip() == ''):
                #print(tag.tag, tag.attrib, tag.attrib['type'], "|", dele)
                dele.append(tag)

        if tag.tag == "head" and tag.text.strip() == '.':
            dele.append(tag)
    #print("~~~~>", dele)
    for d in dele:
        #print(d, d.attrib)
        if d.tag in ["pb", f"{TEI_NS}pb", f"{XML_NS}pb"] or \
            (d.attrib is not None and \
            'type' in d.attrib and \
            d.attrib['type'] in ['motHeader', "motTitle", "motSubmissionInfo"]):
            pass
        else:
    #        print(d)
            d.getparent().remove(d)
    #print("end of cleanup", [_.tag for _ in root.iter()])
    return root, write


def clean_xml(root):
    root, write = del_empty_elems(root)
    return root, write
