from lxml import etree
import argparse
from pyriksdagen.utils import protocol_iterators, infer_metadata
import pyriksdagen.download as pydl
import progressbar
from os import mkdir

def get_pkg_name(protocol):
    # function to get package name from a xml file name
    # packages used to connect to kb-lab
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.parse(protocol, parser).getroot()
    for elem in root.iter():
        if elem.tag == root.tag[:-3] + 'head':
            pkg_name = elem.text
            break
    return pkg_name

def page_as_string(page_number):
    # changes page number to string format used in kblab database
    return f"{page_number:0>3}"


def main(args):
    
    archive = pydl.LazyArchive()
    protocols = sorted(list(protocol_iterators(args.records_folder, start=args.start, end=args.end)))
    
    curr_year = infer_metadata(protocols[0])['year']
    year_dir = f'{args.save_folder}{curr_year}/'
    mkdir(year_dir)
    for protocol in progressbar.progressbar(protocols):
        next_year = infer_metadata(protocol)['year']
        if curr_year != next_year:
            # switch current folder
            year_dir = f'{args.save_folder}/{next_year}/'
            mkdir(year_dir)
            curr_year = next_year
        # for each protocol make a new folder
        pkg_name = get_pkg_name(protocol)
        pkg = archive.get(pkg_name)
        protocol_dir = f'{year_dir}{pkg_name}/'
        mkdir(protocol_dir)
        file_list = pkg.list()
        xml_list = [file for file in file_list if file[-3:] == 'xml']
        for alto_file in xml_list:
            alto_response = pkg.get_raw(alto_file)
            with open(f'{protocol_dir}{alto_file}', 'wb') as foutput:
                for line in alto_response.readlines():
                    foutput.write(line)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records_folder", type=str, default="corpus/records")
    parser.add_argument("--save_folder", type=str)
    parser.add_argument("-s", "--start", type=int, default=None, help="Start year")
    parser.add_argument("-e", "--end", type=int, default=None, help="End year")
    args = parser.parse_args()
    main(args)