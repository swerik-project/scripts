from lxml import etree
import argparse
from pyriksdagen.utils import elem_iter, protocol_iterators, infer_metadata, XML_NS
import re
import progressbar
import pandas as pd

####    utility functions   ####
def cleanup_segments(text_seq):
    # function to remove whitespace from string to get comparable text between corpus and kblab
    text_seq = text_seq.split()
    text_seq_list = [s for s in text_seq if s != '']
    text_seq_string = ' '.join(text_seq_list)
    return text_seq_string

####    functions to create context sequences from multiple strings ####
def concat_sequences(previous_sequence, current_sequence, next_sequence, context_type, target_length = 120):
    # if previous sequence is long, we want it to truncate the sequence so that the
    # current sequence is not unecessarily 
    if context_type == 'left_context':
        max_previous_length = target_length//2
    elif context_type == 'full_context':
        max_previous_length = target_length//3
    
    # cleaning up sequences
    previous_sequence = cleanup_segments(str(previous_sequence))
    current_sequence = cleanup_segments(str(current_sequence))
    next_sequence = cleanup_segments(str(next_sequence))
    
    
    previous_as_list = re.split(r'([.!?])', previous_sequence)
    if (previous_as_list[-1] == '') & (len(previous_as_list) != 1):
        prev_last_sentence = previous_as_list[-3:]
        prev_last_sentence = ''.join(prev_last_sentence)
    else:
        prev_last_sentence = previous_as_list[-1]
        
    next_as_list = re.split(r'([.!?])', next_sequence)
    if len(next_as_list) != 1:
        next_first_sentence = next_as_list[:2]
        next_first_sentence = ''.join(next_first_sentence)
    else:
        next_first_sentence = next_as_list[0]

    # regardless of sequence type, we combine prev last sentence with curr sequence
    prev_last_sentence_as_list = prev_last_sentence.split(' ')
    n_words = len(prev_last_sentence_as_list)
    if n_words > max_previous_length:
        prev_last_sentence_as_list = prev_last_sentence_as_list[-max_previous_length:]
        prev_last_sentence = ' '.join(prev_last_sentence_as_list)
    # use new line (/n) as token to signify where current sequence begins
    left_context_sequence = prev_last_sentence + ' /n ' + current_sequence
    
    if context_type == 'left_context':
        return left_context_sequence
    elif context_type == 'full_context':
        # add next first sentence to left context sequence to get full context
        full_context_sequence = left_context_sequence + ' /n ' + next_first_sentence
        return full_context_sequence


def get_context_sequence(protocol, context_type, max_length = 120):
    id_list = []
    context_sequence_list = []
    
    id_key = XML_NS + 'id'
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.parse(protocol, parser).getroot()
    
    prev_elem_is_text_seq = False
    elem_idx = ''
    prev_sequence = ''
    next_sequence = ''
    prev_elem_sequence = ''
    for tag, elem in elem_iter(root):
        if tag == 'note':
            elem_sequence = elem.text
            elem_idx = elem.attrib[id_key]
            
            if prev_elem_is_text_seq == True:
                next_sequence = elem_sequence
                context_sequence = concat_sequences(prev_sequence, curr_sequence, next_sequence, context_type, max_length)
                
                id_list.append(idx)
                context_sequence_list.append(context_sequence)
            

            idx = elem_idx
            curr_sequence = elem_sequence
            prev_sequence = prev_elem_sequence
                
            prev_elem_sequence = elem_sequence
            prev_elem_is_text_seq = True
        elif tag == 'u':
            for child in elem.getchildren():
                elem_sequence = child.text
                elem_idx = child.values()[0]
                
                if prev_elem_is_text_seq == True:
                    next_sequence = elem_sequence
                    context_sequence = concat_sequences(prev_sequence, curr_sequence, next_sequence, context_type, max_length)
                    
                    id_list.append(idx)
                    context_sequence_list.append(context_sequence)
                    
                
                idx = elem_idx
                curr_sequence = elem_sequence
                prev_sequence = prev_elem_sequence
                    
                prev_elem_sequence = elem_sequence
                prev_elem_is_text_seq = True
                
    next_sequence = ''
    context_sequence = concat_sequences(prev_sequence, curr_sequence, next_sequence, context_type, max_length)
    
    id_list.append(idx)
    context_sequence_list.append(context_sequence)
    
    
    output_dict = {'id' : id_list,
                   'context_sequence' : context_sequence_list}
    return output_dict

def main(args):
    
    protocols = sorted(list(protocol_iterators(args.records_folder, start=args.start, end=args.end)))
    
    curr_year = infer_metadata(protocols[0])['year']
    
    context_sequence_dict = {'id' : [],
                             'context_sequence' : []}
    
    for protocol in progressbar.progressbar(protocols):
        next_year = infer_metadata(protocol)['year']
        
        if curr_year != next_year:
            output_df = pd.DataFrame.from_dict(context_sequence_dict)
            save_file = args.save_folder + str(curr_year) + '_' + args.context_type + '.csv'
            output_df.to_csv(save_file, index = False)
            
            # reset pooling dict
            context_sequence_dict = {'id' : [],
                             'context_sequence' : []}
            curr_year = next_year
        
        protocol_context_sequence_dict = get_context_sequence(protocol, args.context_type)

        context_sequence_dict['id'].extend(protocol_context_sequence_dict['id'])
        context_sequence_dict['context_sequence'].extend(protocol_context_sequence_dict['context_sequence'])
    
    output_df = pd.DataFrame.from_dict(context_sequence_dict)
    save_file = args.save_folder + str(curr_year) + '_' + args.context_type + '.csv'
    output_df.to_csv(save_file, index = False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records_folder", type=str, default="corpus/records")
    parser.add_argument("--save_folder", type=str)
    parser.add_argument("-s", "--start", type=int, default=None, help="Start year")
    parser.add_argument("-e", "--end", type=int, default=None, help="End year")
    parser.add_argument("--context_type", type=str, choices = ['left_context', 'full_context'])
    args = parser.parse_args()
    main(args)