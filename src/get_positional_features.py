from lxml import etree
import argparse
from pyriksdagen.utils import elem_iter, protocol_iterators
import pyriksdagen.download as pydl
from difflib import SequenceMatcher
import pandas as pd
import progressbar

# file which code to get positional features will be added
#### functions to get position features from xml files ####
def get_chamber(protocol):
    protocol_dir_list = protocol.split('-')
    
    if 'ak' in protocol_dir_list:
        return 'ak'
    elif 'fk' in protocol_dir_list:
        return 'fk'
    else:
        return 'unicameral'
    
def second_chamber_bin(chamber):
    if chamber == 'ak':
        return 1
    else:
        return 0
def unicameral_bin(chamber):
    if chamber == 'unicameral':
        return 1
    else:
        return 0
    
def even_or_odd(x):
        if x & 1 == 0:
            return 1
        else:
            return 0
        
def relative_page_number(x, max_x):
    try:
        output = (x/max_x)*1000
    except:
        output = 0
    return output

def cleanup_segments(text_seq):
    # function to remove whitespace from string to get comparable text between corpus and kblab
    text_seq = text_seq.translate({ord('\n'): ' '})
    text_seq = text_seq.split(' ')
    text_seq_list = [s for s in text_seq if s != '']
    text_seq_string = ' '.join(text_seq_list).strip()
    return text_seq_string

def get_positional_features(protocol):
    # function to get positional features which can be parsed from xml-files
    
    id_key = '{http://www.w3.org/XML/1998/namespace}id'
    year_str = protocol.split('\\')[-2][:4]
    year = int(year_str)
    chamber = get_chamber(protocol)

    id_list = []
    page_number_list = []
    even_page_list = []
    year_list = []
    chamber_list = []
    
    # used in next function to get position coordinates
    text_list = []
    intro_speech_list = []

    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.parse(protocol, parser).getroot()
    for tag, elem in elem_iter(root):
        # pb tags contain link to page image which also indicates page number
        if tag == 'pb':
            page_number_str = elem.attrib['facs'].split('.jp2')[0].split('-')[-1]
            page_number = int(page_number_str)
        # only note and seg tags contain text sequences
        # seg tags nested in u tags
        elif tag == 'note':
            elem_id = elem.attrib[id_key]
            id_list.append(elem_id)
            page_number_list.append(page_number)
            even_page_list.append(even_or_odd(page_number))
            year_list.append(year)
            chamber_list.append(chamber)
            
            text_list.append(cleanup_segments(elem.text))
            if 'speaker' in elem.attrib.values():
                intro_speech_list.append(1)
            else:
                intro_speech_list.append(0)
                
        elif tag == 'u':
            for child in elem.getchildren():
                elem_id = child.attrib[id_key]
                id_list.append(elem_id)
                page_number_list.append(page_number)
                even_page_list.append(even_or_odd(page_number))
                year_list.append(year)
                chamber_list.append(chamber)
                
                text_list.append(cleanup_segments(child.text))
                intro_speech_list.append(0)
                
                
    second_chamber_list = [second_chamber_bin(x) for x in chamber_list]
    unicameral_list = [unicameral_bin(x) for x in chamber_list]
    relative_page_number_list = [relative_page_number(x, page_number) for x in page_number_list]
    
    output_dict = {'id' : id_list,
                   'relative_page_number': relative_page_number_list,
                   'year' : year_list,
                   'even_page' : even_page_list,
                   'second_chamber' : second_chamber_list,
                   'unicameral' : unicameral_list,
                   'text' : text_list,
                   'intro_speech' : intro_speech_list,
                   'page_number' : page_number_list}
    
    return output_dict

####   functions to get coords from alto files ####

def get_pkg_name(file):
    # function to get package name from a xml file name
    # packages used to connect to kb-lab
    file_list = file.split('\\')
    file_name = file_list[-1].rstrip('.xml')
    year = int(file_name[5:9])
    if (year > 1874) & (file_name[-1] != '-'):
        if (len(file_name) == 22) & (file_name.split('-')[2] == 'a'):
            f_number = str(int(file_name[-6:-3]))
            pkg_name = file_name[:-6] + f_number + file_name[-3:]
        elif file_name[-3:] in ['-01', '-02']:
            f_number = str(int(file_name.split('-')[5]))
            if file_name[-3:] == '-01':
                pkg_name = file_name[:15] + f_number
            else:
                pkg_name = file_name[:15] + f_number + 'a'
        else:
            f_number = str(int(file_name[-3:]))
            pkg_name = file_name[:-3] + f_number
    
    else:
        pkg_name = file_name
    return pkg_name

def get_pkg(pkg_id, archive):
    # connect to a package in kblab
    pkg = archive.get(pkg_id)
    # handle special case where protocol_id does not match id in kb-lab
    add_to_string = ['höst', 'extrahöst']
    if pkg == None:
        for s in add_to_string:
            p_list = pkg_id.split('-')
            p_list[2] = s
            new_id = ''
            for p in p_list:
                new_id += p
                new_id += '-'
            pkg = archive.get(new_id[:-1])
            pkg_id = new_id[:-1]
            if pkg != None:
                break
    return pkg, pkg_id

def page_as_string(page_number):
    # changes page number to string format used in kblab database
    s = str(page_number)
    if len(s) == 1:
        s = '00' + s
    elif len(s) == 2:
        s = '0' + s
    return s

def update_block_position(prev_pos, curr_pos):
    output_0 = min(prev_pos[0], curr_pos[0])
    output_1 = min(prev_pos[1], curr_pos[1])
    output_2 = max(prev_pos[2], curr_pos[2])
    output_3 = max(prev_pos[3], curr_pos[3])
    
    return (output_0, output_1, output_2, output_3)

def get_img_box(input_box):
    # returns corner coordinates of img as tuple
    input_box = [float(coord) for coord in input_box]
    output = (input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3])
    return output

def string_similarity(a, b):
    # computes how similar two strings are
    # used for QC, low similarity requires manual review of 2d position
    return SequenceMatcher(None, a, b).ratio()

def get_page_xml(pkg, pkg_name, page_number):
    parser = etree.XMLParser(remove_blank_text=True)
    
    # Function searches alto files for width and height of a page in a protocol
    pkg_name = pkg_name.replace('-', '_')
    
    xml_id = pkg_name + '-' + page_as_string(page_number) + '.xml'
    pkg_xml = pkg.get_raw(xml_id)
    tree = etree.parse(pkg_xml, parser).getroot()
    
    return tree

def matching_index(l, x):
    return [i for i, n in enumerate(l) if n == x]

def get_page_pos(pkg, pkg_name, page_number, target_sequences, target_is_intro_speech):
    # all sequences on page need to be supplied to function
    # sequences need to be supplied in order
    
    try:
        xml_tree = get_page_xml(pkg, pkg_name, page_number)
    except:
        null_data = [None for x in target_sequences]
        matched_sequences = null_data
        matched_positions = null_data
        matched_page_size = null_data
        matched_similarities = null_data
        
        return matched_sequences, matched_positions, matched_page_size, matched_similarities
    #output
    matched_sequences = []
    matched_positions = []
    matched_similarities = []
    
    # vars needed for loop
    block_sequences = []
    curr_sequence_list = []
    curr_pos = (float('inf'), float('inf'), 0.0, 0.0)
    block_pos = [9999.0, 9999.0, 0.0, 0.0]
    prev_block = 0
    split_word_count = 0
    
    # loop through elements on page
    for elem in xml_tree.iter():
        attrib = elem.attrib
        if elem.tag[-4:] == 'Page':
            page_width, page_height = int(elem.attrib['WIDTH']), int(elem.attrib['HEIGHT'])
        if 'ID' in attrib.keys():
            if attrib['ID'][:5] == 'block':
                curr_block = int(attrib['ID'].split('_')[1])
                if curr_block != prev_block:
                    curr_sequence = ' '.join(curr_sequence_list).strip()
                    if curr_sequence.rstrip(' ') == '':
                        prev_block = curr_block
                        curr_sequence_list = []
                    else:
                        block_sequences.append(curr_sequence)
                        curr_pos = update_block_position(curr_pos, get_img_box(block_pos))
                        curr_sequence_list = []
                        prev_block = curr_block
                        # match sequence to block if it is same number of characters or similar
                        block_similarity = string_similarity(target_sequences[0], ' '.join(block_sequences))
                        
                        if (len(' '.join(block_sequences).split(' '))-split_word_count) != 0:
                            diff_ratio = abs(1-len(target_sequences[0].split(' '))/(len(' '.join(block_sequences).split(' '))-split_word_count))
                        else:
                            diff_ratio = abs(1-len(target_sequences[0].split(' '))/(len(' '.join(block_sequences).split(' '))-(split_word_count-1)))

                        if (block_similarity > 0.7) | (0.25 >= diff_ratio):
                            matched_sequences.append(' '.join(block_sequences))
                            matched_positions.append(curr_pos)
                            matched_similarities.append(block_similarity)
                            curr_pos = (float('inf'), float('inf'), 0.0, 0.0)
                            target_sequences.pop(0)
                            target_is_intro_speech.pop(0)
                            block_sequences = []
                            split_word_count = 0
                        
                            if len(target_sequences) == 0:
                                break
                        
                block_pos = [attrib['HPOS'], attrib['VPOS'], attrib['WIDTH'], attrib['HEIGHT']]
            elif attrib['ID'][:6] == 'string':
                
                curr_word = attrib['CONTENT'] 
                
                try:
                    attrib['SUBS_CONTENT']
                    if attrib['SUBS_TYPE'] == 'HypPart1':
                        split_word_count += 1
                except:
                    if curr_word[-1] == '-':
                        split_word_count += 1
                
                if curr_word.rstrip(' ') == '':
                    pass
                else:
                    curr_sequence_list.append(curr_word)
                
                if target_is_intro_speech[0] == 1:
                    word_pos = [attrib['HPOS'], attrib['VPOS'], attrib['WIDTH'], attrib['HEIGHT']]
                    curr_pos = update_block_position(curr_pos, get_img_box(word_pos))
                    if ':' in curr_word:
                        curr_sequence = ' '.join(curr_sequence_list)
                        curr_colon_split = curr_sequence.split(':')
                        curr_seq_0 = curr_colon_split[0]
                        curr_seq_1 = ':'.join(curr_colon_split[1:])
                        
                        matched_sequences.append(curr_seq_0 + ':')
                        matched_positions.append(curr_pos)
                        matched_similarities.append(string_similarity(target_sequences[0], curr_seq_0))
                        curr_pos = get_img_box(word_pos)
                        target_sequences.pop(0)
                        target_is_intro_speech.pop(0)
                        
                        curr_sequence_list = curr_seq_1.split(' ')
        
        if len(target_sequences) == 0:
            break
    
    # add match for last sequence
    if len(target_sequences) != 0:
        curr_sequence = ' '.join(curr_sequence_list)
        block_sequences.append(curr_sequence)   
        block_sequence = ' '.join(block_sequences)
        matched_sequences.append(block_sequence)
        matched_positions.append(update_block_position(curr_pos, get_img_box(block_pos)))
        block_similarity = string_similarity(target_sequences[0], block_sequence)
        matched_similarities.append(block_similarity)
        target_sequences.pop(0)
    # add width and height
    matched_page_size = [(page_width, page_height) for x in range(len(matched_sequences))]
    # return sequences from alto files and positions to verify match
    return matched_sequences, matched_positions, matched_page_size, matched_similarities

def add_coord_to_dict(protocol, pos_dict, archive):
    pkg_name = get_pkg_name(protocol)
    pkg, pkg_name = get_pkg(pkg_name, archive)
    
    pos_lefts = []
    pos_uppers = []
    pos_rights = []
    pos_lowers = []
    widths = []
    heights = []
    similarities = []
    
    # iterate through each page of the protocol
    page_numbers = pos_dict['page_number']
    unique_page_numbers = set(page_numbers)
    for page_number in unique_page_numbers:
        indices = matching_index(page_numbers, page_number)
        
        # input to get_page_pos() used to get coordinates
        target_sequences = pos_dict['text'][indices[0]:(indices[-1]+1)]
        target_is_intro_speech = pos_dict['intro_speech'][indices[0]:(indices[-1]+1)]
        
        # get coordinates and other features 
        page_text, page_coord, page_width_height, page_similarities = get_page_pos(pkg, pkg_name, page_number, target_sequences, target_is_intro_speech)

        page_pos_left = [x[0] for x in page_coord]
        page_pos_upper = [x[1] for x in page_coord]
        page_pos_right = [x[2] for x in page_coord]
        page_pos_lower = [x[3] for x in page_coord]
        page_width = [x[0] for x in page_width_height]
        page_height = [x[1] for x in page_width_height]
        
        # add outputs to list
        pos_lefts.extend(page_pos_left)
        pos_uppers.extend(page_pos_upper)
        pos_rights.extend(page_pos_right)
        pos_lowers.extend(page_pos_lower)
        widths.extend(page_width)
        heights.extend(page_height)
        similarities.extend(page_similarities)
        
    # add outputs to dict
    pos_dict['posLeft'] = pos_lefts
    pos_dict['posUpper'] = pos_uppers
    pos_dict['posRight'] = pos_rights
    pos_dict['posLower'] = pos_lowers
    pos_dict['width'] = widths
    pos_dict['height'] = heights
    pos_dict['similarities'] = similarities
    
    # return dict with all features
    return pos_dict

def main(args):
    
    feature_dict = {'id': [],
                'record': [],
                'page_number': [],
                'posLeft' : [],
                'posUpper' : [],
                'posRight' : [],
                'posLower' : [],
                'relative_page_number' : [],
                'even_page' : [],
                'year' : [],
                'second_chamber' : [],
                'unicameral' : [], 
                'width' : [],
                'height' : []}
    
    archive = pydl.LazyArchive()
    protocols = sorted(list(protocol_iterators(args.records_folder, start=args.start, end=args.end)))
    
    curr_year = protocols[0].split('\\')[-2]
    for protocol in progressbar.progressbar(protocols):
        
        next_year = protocol.split('\\')[-2]
        if curr_year != next_year:
            output_df = pd.DataFrame.from_dict(feature_dict)
            save_file = curr_year + '_position_features.csv'
            output_df.to_csv(save_file, index = False)

            curr_year = next_year
            
            feature_dict = {'id': [],
                'record': [],
                'page_number': [],
                'posLeft' : [],
                'posUpper' : [],
                'posRight' : [],
                'posLower' : [],
                'relative_page_number' : [],
                'even_page' : [],
                'year' : [],
                'second_chamber' : [],
                'unicameral' : [], 
                'width' : [],
                'height' : []}
            
        # get xml position features
        protocol_feature_dict = get_positional_features(protocol)
        
        # add coordinate data
        protocol_feature_dict = add_coord_to_dict(protocol, protocol_feature_dict, archive)

        # add to feature dict
        record_file = protocol.split('\\')[-1]
        feature_dict['id'].extend(protocol_feature_dict['id'])
        feature_dict['record'].extend([record_file for x in range(len(protocol_feature_dict['id']))])
        feature_dict['page_number'].extend(protocol_feature_dict['page_number'])
        feature_dict['posLeft'].extend(protocol_feature_dict['posLeft'])
        feature_dict['posUpper'].extend(protocol_feature_dict['posUpper'])
        feature_dict['posRight'].extend(protocol_feature_dict['posRight'])
        feature_dict['posLower'].extend(protocol_feature_dict['posLower'])
        feature_dict['relative_page_number'].extend(protocol_feature_dict['relative_page_number'])
        feature_dict['even_page'].extend(protocol_feature_dict['even_page'])
        feature_dict['year'].extend(protocol_feature_dict['year'])
        feature_dict['second_chamber'].extend(protocol_feature_dict['second_chamber'])
        feature_dict['unicameral'].extend(protocol_feature_dict['unicameral'])
        feature_dict['width'].extend(protocol_feature_dict['width'])
        feature_dict['height'].extend(protocol_feature_dict['height'])
    
    # store features in dataframe and save to disk
    save_file = curr_year + '_position_features.csv'
    output_df = pd.DataFrame.from_dict(feature_dict)
    output_df.to_csv(save_file, index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records_folder", type=str, default="corpus/records")
    parser.add_argument("-s", "--start", type=int, default=None, help="Start year")
    parser.add_argument("-e", "--end", type=int, default=None, help="End year")
    args = parser.parse_args()
    main(args)