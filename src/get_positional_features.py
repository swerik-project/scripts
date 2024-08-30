from lxml import etree
import argparse
#from pyriksdagen.utils import elem_iter, protocol_iterators, infer_metadata, XML_NS
import pyriksdagen.download as pydl
from difflib import SequenceMatcher
import pandas as pd
import progressbar
import pyriksdagen.download as pydl
from pyriksdagen.utils import remove_whitespace_from_sequence, elem_iter, XML_NS, infer_metadata
from alto import parse_file
from alto import String as altoString
from thefuzz import fuzz
import re

#### functions to get position features from xml files ####
    
def even_or_odd(x):
        if x % 2 == 0:
            return 1
        else:
            return 0
        
def relative_page_number(x, max_x):
    try:
        output = (x/max_x)*1000
    except:
        output = 0.0
    return output

def cleanup_segments(text_seq):
    # function to remove whitespace from string to get comparable text between corpus and kblab
    text_seq = text_seq.split()
    text_seq_list = [s for s in text_seq if s != '']
    text_seq_string = ' '.join(text_seq_list)
    return text_seq_string

def get_positional_features(protocol):
    # function to get positional features which can be parsed from xml-files
    
    id_key = f"{XML_NS}id"
    meta_data = infer_metadata(protocol)
    year = meta_data['year']
    chamber = meta_data['chamber']

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
    page_number = 0
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
                
                
    second_chamber_list = [1 if chamber == 'Andra kammaren' else 0 for chamber in chamber_list]
    unicameral_list = [1 if chamber == 'Enkammarriksdagen' else 0 for chamber in chamber_list]
    relative_page_number_list = [relative_page_number(x, page_number) for x in page_number_list]
    
    output_dict = {'id' : id_list,
                   'relative_page_number': relative_page_number_list,
                   'year' : year_list,
                   'even_page' : even_page_list,
                   'is_second_chamber' : second_chamber_list,
                   'is_unicameral' : unicameral_list,
                   'text' : text_list,
                   'intro_speech' : intro_speech_list,
                   'page_number' : page_number_list}
    
    return output_dict

####   functions to get coords from alto files ####

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


def string_similarity(a, b):
    # computes how similar two strings are
    # used for QC, low similarity requires manual review of 2d position
    return SequenceMatcher(None, a, b).ratio()

def matching_index(l, x):
    return [i for i, n in enumerate(l) if n == x]

def get_page_pos(pkg, pkg_name, alto_folder, page_number, target_sequences, target_is_intro_speech, page_number_offset = 0):
    # all sequences on page need to be supplied to function
    # sequences need to be supplied in order
    page_found = False
    
    while page_found == False:
        if alto_folder != None:
            try:
                alto_file = f'{alto_folder}-{page_as_string(page_number+page_number_offset)}.xml'
                alto = parse_file(alto_file)
                page_found = True
            except:
                page_number_offset += 1
                if page_number_offset == 20:
                    # treat as null data
                    matched_sequences = [None for x in target_sequences]
                    matched_positions = [(None, None, None, None) for x in target_sequences]
                    matched_similarities = [None for x in target_sequences]
                    matched_page_size = [(None, None) for x in target_sequences]
                    return matched_sequences, matched_positions, matched_page_size, matched_similarities, 0
        else:
            try:
                pkg_name = pkg_name.replace('-', '_')
                xml_id = pkg_name + '-' + page_as_string(page_number) + '.xml'
                alto = parse_file(pkg.get_raw(xml_id))
                page_found = True
            except:
                page_number_offset += 1
                if page_number_offset == 20:
                    # treat as null data
                    matched_sequences = [None for x in target_sequences]
                    matched_positions = [(None, None, None, None) for x in target_sequences]
                    matched_similarities = [None for x in target_sequences]
                    matched_page_size = [(None, None) for x in target_sequences]
                    return matched_sequences, matched_positions, matched_page_size, matched_similarities, 0
                
    #output
    matched_sequences = []
    matched_positions = []
    matched_similarities = []
    
    # add width and height
    page_info = alto.layout.pages[0]
    page_width, page_height = page_info.width, page_info.height
    matched_page_size = [(page_width, page_height) for x in range(len(target_sequences))]

    # vars needed for loop
    curr_sequence_list = []
    curr_pos = (float('inf'), float('inf'), 0.0, 0.0)
    block_pos = [9999.0, 9999.0, 0.0, 0.0]
    n_split_words = 0
    n_colons_in_curr_sequence = 0
    

    for block in alto.extract_text_blocks():
        # we need to treat intro speeches differently since they do not match text blocks
        if target_is_intro_speech[0] == 1:
            n_target_colons = target_sequences[0].count(':')
            n_colons_in_curr_sequence = 0
            for text_line in block.text_lines:
                for string in text_line.strings:
                    if str(type(string)) == "<class 'alto.String'>":
                        curr_word = string.content
                        if curr_word.strip() == '':
                            # ignore blank spaces
                            continue
                        word_pos = get_img_box([string.hpos, string.vpos, string.width, string.height])
                        curr_pos = update_block_position(curr_pos, word_pos)
                        # add to list
                        curr_sequence_list.append(curr_word)
                        if ':' in curr_word:
                            n_colons_in_curr_sequence += 1
                        elif curr_word[-1] == '-':
                            n_split_words += 1
                        if (n_target_colons == n_colons_in_curr_sequence) & (target_is_intro_speech[0] == 1) & (n_target_colons != 0):
                            curr_sequence = ' '.join(curr_sequence_list)
                            curr_similarity = string_similarity(curr_sequence, target_sequences[0])
                            matched_sequences.append(curr_sequence)
                            matched_positions.append(curr_pos)
                            matched_similarities.append(curr_similarity)
                            # reset vars for next match
                            curr_pos = (float('inf'), float('inf'), 0.0, 0.0)
                            curr_sequence_list = []
                            n_split_words = 0
                            n_colons_in_curr_sequence = 0
                            target_sequences.pop(0)
                            target_is_intro_speech.pop(0)
                        elif (n_target_colons == 0) & (len(curr_sequence_list) == (len(target_sequences[0].split())-n_split_words)):
                            curr_sequence = ' '.join(curr_sequence_list)
                            curr_similarity = string_similarity(curr_sequence, target_sequences[0])
                            matched_sequences.append(curr_sequence)
                            matched_positions.append(curr_pos)
                            matched_similarities.append(curr_similarity)
                            # reset vars for next match
                            curr_pos = (float('inf'), float('inf'), 0.0, 0.0)
                            curr_sequence_list = []
                            n_split_words = 0
                            n_colons_in_curr_sequence = 0
                            target_sequences.pop(0)
                            target_is_intro_speech.pop(0)
                        if len(target_sequences) == 0:
                            return matched_sequences, matched_positions, matched_page_size, matched_similarities, page_number_offset
      
            # check if remaining words in block are a sequence or not
            curr_sequence = ' '.join(curr_sequence_list)
            curr_similarity = string_similarity(curr_sequence, target_sequences[0])
            
            target_sequence_length = len(target_sequences[0].split())
            curr_sequence_length = len(curr_sequence_list)    
            if (curr_sequence_length - n_split_words) != 0:
                diff_ratio = abs(1 - target_sequence_length/(curr_sequence_length-n_split_words))
            else:
                diff_ratio = abs(1 - target_sequence_length/(curr_sequence_length - n_split_words + 1))
                
            if (curr_similarity > 0.7) | (0.25 >= diff_ratio):
                matched_sequences.append(curr_sequence)
                matched_positions.append(curr_pos)
                matched_similarities.append(curr_similarity)
                # reset vars for next match
                curr_pos = (float('inf'), float('inf'), 0.0, 0.0)
                curr_sequence_list = []
                n_split_words = 0
                target_sequences.pop(0)
                target_is_intro_speech.pop(0)
            # exit code if all sequences found
            if len(target_sequences) == 0:
                return matched_sequences, matched_positions, matched_page_size, matched_similarities, page_number_offset
                
        else:
            block_sequence_list = block.extract_words()
            if ' '.join(block_sequence_list).strip() == '': 
                # ignore empty blocks
                continue
            block_pos = get_img_box([block.hpos, block.vpos, block.width, block.height])
            # count number of split words in block and add to total
            n_split_words += sum([1 if word[-1] == '-' else 0 for word in block_sequence_list])
            
            curr_sequence_list.extend(block_sequence_list)
            curr_pos = update_block_position(curr_pos, block_pos)
            
            curr_sequence = ' '.join(curr_sequence_list)
            
            # compute string similarity and difference in length to check for match
            curr_similarity = string_similarity(curr_sequence, target_sequences[0])
            target_sequence_length = len(target_sequences[0].split())
            curr_sequence_length = len(curr_sequence_list)    
            if (curr_sequence_length - n_split_words) != 0:
                diff_ratio = abs(1 - target_sequence_length/(curr_sequence_length-n_split_words))
            else:
                diff_ratio = abs(1 - target_sequence_length/(curr_sequence_length - n_split_words + 1))
            
            # if close enough match, add to output
            if (curr_similarity > 0.7) | (0.25 >= diff_ratio):
                matched_sequences.append(curr_sequence)
                matched_positions.append(curr_pos)
                matched_similarities.append(curr_similarity)
                # reset vars for next match
                curr_pos = (float('inf'), float('inf'), 0.0, 0.0)
                curr_sequence_list = []
                n_split_words = 0
                target_sequences.pop(0)
                target_is_intro_speech.pop(0)
                
                # exit code if all sequences found
                if len(target_sequences) == 0:
                    return matched_sequences, matched_positions, matched_page_size, matched_similarities, page_number_offset
            
    # add last sequence
    if len(target_sequences) != 0:
        curr_sequence = ' '.join(curr_sequence_list)
        matched_sequences.append(curr_sequence)
        matched_positions.append(curr_pos)
        matched_similarities.append(string_similarity(target_sequences[0], curr_sequence))

    # return sequences from alto files and positions to verify match
    return matched_sequences, matched_positions, matched_page_size, matched_similarities, page_number_offset

def add_coord_to_dict(protocol, pos_dict, alto_folder = None, archive = None):   
    protocol_year = infer_metadata(protocol)['year']
    protocol_name = get_pkg_name(protocol)
    # if alto_folder is not provided, get data from kblab
    if alto_folder == None:
        pkg_name = get_pkg_name(protocol)
        pkg = archive.get(pkg_name)
    else:
        alto_folder = f"{alto_folder}{protocol_year}/{protocol_name}/{protocol_name.replace('-', '_')}"
        pkg = None
        pkg_name = None
    pos_lefts = []
    pos_uppers = []
    pos_rights = []
    pos_lowers = []
    widths = []
    heights = []
    similarities = []
    offsets = []
    
    # iterate through each page of the protocol
    page_numbers = pos_dict['page_number']
    unique_page_numbers = sorted(set(page_numbers))
    page_number_offset = 0
    n_failed_pages = 0
    for page_number in unique_page_numbers:
        indices = matching_index(page_numbers, page_number)
        
        # get coordinates and other features 
        correct_matched_length = False
        while correct_matched_length == False:
            # input to get_page_pos() used to get coordinates
            target_sequences = pos_dict['text'][indices[0]:(indices[-1]+1)]
            target_is_intro_speech = pos_dict['intro_speech'][indices[0]:(indices[-1]+1)]
            page_text, page_coord, page_width_height, page_similarities, page_number_offset = get_page_pos(pkg, pkg_name, alto_folder, page_number, 
                                                                                                       target_sequences, target_is_intro_speech,
                                                                                                       page_number_offset)
         
            if len(page_text) == len(pos_dict['text'][indices[0]:(indices[-1]+1)]):
                correct_matched_length = True
            else:
                page_number_offset += 1
                if (page_number_offset > 20) | (n_failed_pages > 5):
                    # restart and treat page as null data
                    correct_matched_length = True
                    page_number_offset = 0
                    page_coord = [(None, None, None, None) for x in range(len(pos_dict['text'][indices[0]:(indices[-1]+1)]))]
                    page_width_height = [(None, None) for x in range(len(pos_dict['text'][indices[0]:(indices[-1]+1)]))]
                    page_similarities = [None for x in range(len(pos_dict['text'][indices[0]:(indices[-1]+1)]))]
                    n_failed_pages += 1
                    
                    
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
        offsets.extend([page_number_offset for x in page_pos_left])
        
    # add outputs to dict
    pos_dict['posLeft'] = pos_lefts
    pos_dict['posUpper'] = pos_uppers
    pos_dict['posRight'] = pos_rights
    pos_dict['posLower'] = pos_lowers
    pos_dict['width'] = widths
    pos_dict['height'] = heights
    pos_dict['similarities'] = similarities
    pos_dict['page_offset'] = offsets
    
    # return dict with all features
    return pos_dict

def get_max_index(l, elem):
    """
    If there are identical text sequences on a page, returns index of sequence which is closest in sequential order
    """
    maxval = None
    for i, val in enumerate(l):
        if maxval is None or val > maxval:
            indices = [i]
            maxval = val
        elif val == maxval:
            indices.append(i)
    if len(indices) == 1:
        return indices[0]
    else:
        previous_elements = elem.xpath("preceding::*[local-name() = 'note' or local-name() = 'seg' or local-name() = 'pb']")
        for i, e in enumerate(reversed(previous_elements)):
            if e.tag[-2:] == 'pb':
                page_position = i
                break
        return min(indices, key = lambda x: abs(x - page_position))

def merge_dictionaries(dict1, dict2):
    output_dict = {**dict1, **dict2}
    return output_dict

def get_img_box(input_box):
    # returns corner coordinates of img as tuple
    input_box = [float(coord) for coord in input_box]
    output = (input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3])
    return output

def update_block_position(prev_pos, curr_pos):
    output_0 = min(prev_pos[0], curr_pos[0])
    output_1 = min(prev_pos[1], curr_pos[1])
    output_2 = max(prev_pos[2], curr_pos[2])
    output_3 = max(prev_pos[3], curr_pos[3])
    
    return (output_0, output_1, output_2, output_3)

def get_protocol_information(elem):
    sequence_id = elem.get(f'{XML_NS}id')
    head_elem = elem.xpath("preceding::*[local-name() = 'head'][1]")[0]
    pkg_name = head_elem.text
    metadata = infer_metadata(pkg_name)
    record_year = metadata['year']
    record_chamber = metadata['chamber']
    page_elem = elem.xpath("preceding::*[local-name() = 'pb'][1]")[0]
    last_page_elem_list = elem.xpath("following::*[local-name() = 'pb']")
    last_page_elem = last_page_elem_list[-1] if len(last_page_elem_list) != 0 else None
    try:
        page_number_as_str = page_elem.get('facs').split('.')[-2][-3:]
        page_number = int(page_number_as_str)
        protocol_length_as_str = last_page_elem.get('facs').split('.')[-2][-3:] if last_page_elem is not None else page_number_as_str
    except:
        page_number_as_str = page_elem.get('facs').split('=')[-1]
        page_number = int(page_number_as_str)
        protocol_length_as_str = last_page_elem.get('facs').split('=')[-1] if last_page_elem is not None else page_number_as_str
    even_page = 1 if page_number % 2 == 0 else 0
    
    output_dict = {'id': sequence_id, 
                   'page_number' : page_number,
                   'pn_as_str' : page_number_as_str,
                   'relative_page_number' : relative_page_number(page_number, int(protocol_length_as_str)),
                   'even_page' : even_page,
                   'record_length' : int(protocol_length_as_str),
                   'year' : record_year,
                   'chamber' : record_chamber,
                   'second_chamber' : 1 if record_chamber == 'Andra kammaren' else 0,
                   'unicameral' : 1 if record_chamber == 'Enkammarriksdagen' else 0,
                   'pkg_name': pkg_name}
    return output_dict

def generate_strings_from_alto_block(alto_block):
    return (string for text_line in alto_block.text_lines 
            for string in text_line.strings 
            if isinstance(string, altoString))

def get_page_position_information(elem, alto, elem_type = None):
    text_sequence = remove_whitespace_from_sequence(elem.text)
    page_info = alto.layout.pages[0]
    page_width, page_height = page_info.width, page_info.height
    text_blocks = []
    alto_blocks = alto.extract_text_blocks()
    text_blocks = [' '.join(text_block) for text_block in alto.extract_grouped_words(group_by = 'TextBlock')]
    
    if elem_type == 'speaker':
        n_colons = text_sequence.count(':')
        text_blocks = [''.join(re.split('(:)', x)[:(n_colons+1)]) for x in text_blocks]
        dist_ratio_list = [fuzz.ratio(text_sequence, x) for x in text_blocks]
        max_index = get_max_index(dist_ratio_list, elem)
        matched_block = alto_blocks[max_index]
        curr_pos = [9999.0, 9999.0, 0.0, 0.0]
        n_passed_colons = 0
        for string in generate_strings_from_alto_block(matched_block):
            curr_word = string.content
            word_pos = get_img_box([string.hpos, string.vpos, string.width, string.height])
            curr_pos = update_block_position(curr_pos, word_pos)
            if ':' in curr_word:
                n_passed_colons += 1
                if n_passed_colons == n_colons:
                    break
        block_pos = curr_pos
    else:
        dist_ratio_list = [fuzz.ratio(text_sequence, x) for x in text_blocks]
        max_index = get_max_index(dist_ratio_list, elem)
        matched_block = alto_blocks[max_index]
        block_pos = get_img_box([matched_block.hpos, matched_block.vpos, matched_block.width, matched_block.height])

    output_dict = {'page_width' : page_width,
                   'page_height' : page_height,
                   'actualposLeft' : block_pos[0],
                   'actualposUpper' : block_pos[1],
                   'actualposRight' : block_pos[2],
                   'actualposLower' : block_pos[3],
                   'posLeft' : (block_pos[0]/page_width)*1000,
                   'posUpper' : (block_pos[1]/page_height)*1000,
                   'posRight' : (block_pos[2]/page_width)*1000,
                   'posLower' : (block_pos[3]/page_height)*1000}
    return output_dict

def page_as_string(page_number):
    # changes page number to string format used in kblab database
    return f"{page_number:0>3}"

def get_elem_data(elem, archive, elem_type = None, page_offset = 0):
    out = get_protocol_information(elem)
    pkg_name = out['pkg_name']
    page_not_found = True
    while page_not_found:
        pn_as_str = page_as_string(out['page_number'] + page_offset)
        xml_id = pkg_name.replace('-', '_') + '-' + pn_as_str + '.xml'
        try:
            pkg = archive.get(pkg_name)
            alto = parse_file(pkg.get_raw(xml_id))
            out_pos = get_page_position_information(elem, alto, elem_type = elem_type)
            page_not_found = False
        except:
            page_offset += 1
            if page_offset >= 11:
                out_pos = {'page_width' : None,
                            'page_height' : None,
                            'posLeft' : None,
                            'posUpper' : None,
                            'posRight' : None,
                            'posLower' : None}
                page_not_found = False
            
    merged_output = merge_dictionaries(out, out_pos)
    return merged_output, page_offset

def return_position_features(protocol, sequence_ids, archive):
    output_dicts = []
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.parse(protocol, parser).getroot()
    page_offset = 0
    for tag, elem in elem_iter(root):
        if tag == 'note':
            elem_id = elem.get(f'{XML_NS}id')
            if elem_id in sequence_ids:
                if 'type' in elem.attrib.keys():
                    elem_type = elem.attrib['type']
                else:
                    elem_type = None
                merged_output, page_offset = get_elem_data(elem, archive, elem_type = elem_type, page_offset = page_offset)
                output_dicts.append(merged_output)
                sequence_ids.remove(elem_id)
        elif tag == 'u':
            for child in elem.getchildren():
                child_id = child.get(f'{XML_NS}id')
                if child_id in sequence_ids:
                    merged_output, page_offset = get_elem_data(child, archive, page_offset = page_offset)
                    output_dicts.append(merged_output)
                    sequence_ids.remove(child_id)
        if len(sequence_ids) == 0:
            break
    return output_dicts


def main(args):
    
    if args.alto_folder == None:
        archive = pydl.LazyArchive()
    else:
        archive = None
    
    feature_dict = {'id': [],
                'record': [],
                'page_number': [],
                'page_offset' : [],
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
    
    protocols = sorted(list(protocol_iterators(args.records_folder, start=args.start, end=args.end)))
    
    curr_year = infer_metadata(protocols[0])['year']
    for protocol in progressbar.progressbar(protocols):
        next_year = infer_metadata(protocol)['year']
        if curr_year != next_year:
            output_df = pd.DataFrame.from_dict(feature_dict)
            save_file = args.save_folder + str(curr_year) + '_position_features.csv'
            output_df.to_csv(save_file, index = False)

            curr_year = next_year
            
            feature_dict = {'id': [],
                'record': [],
                'page_number': [],
                'page_offset' : [],
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
        protocol_feature_dict = add_coord_to_dict(protocol, protocol_feature_dict, alto_folder = args.alto_folder, archive = archive)

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
        feature_dict['page_offset'].extend(protocol_feature_dict['page_offset'])
    
    # store features in dataframe and save to disk
    save_file = args.save_folder + str(curr_year) + '_position_features.csv'
    output_df = pd.DataFrame.from_dict(feature_dict)
    output_df.to_csv(save_file, index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records_folder", type=str, default="corpus/records")
    parser.add_argument("--alto_folder", type = str, default = None)
    parser.add_argument("--save_folder", type=str)
    parser.add_argument("-s", "--start", type=int, default=None, help="Start year")
    parser.add_argument("-e", "--end", type=int, default=None, help="End year")
    args = parser.parse_args()
    main(args)