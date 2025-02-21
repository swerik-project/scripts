import argparse
from pyriksdagen.utils import protocol_iterators, infer_metadata, get_context_sequences_for_protocol
import progressbar
import pandas as pd

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
        
        protocol_context_sequence_dict = get_context_sequences_for_protocol(protocol, args.context_type)

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