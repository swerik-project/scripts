"""
Map introductions to the speaker in the metadata.
"""
import pandas as pd
import argparse
from pyriksdagen.db import load_metadata
from pyriksdagen.refine import redetect_protocol
from pyriksdagen.utils import protocol_iterators, get_data_location
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial




def main(args):
    
    if args.metadata_root is not None:
        metadata_location = args.metadata_root
    else:
        metadata_location = get_data_location("metadata")
    party_mapping, *dfs = load_metadata(metadata_location=metadata_location,
                                        processed_metadata_folder=args.processed_metadata_folder)
    for df in dfs:
        df[["start", "end"]] = df[["start", "end"]].apply(pd.to_datetime, format='%Y-%m-%d')
    metadata = [party_mapping] + dfs

    if args.protocol:
        protocols = [args.protocol]
    else:
        if args.records_folder is not None:
            data_location = args.records_folder
        else:
            data_location = get_data_location("records")
        protocols = sorted(list(protocol_iterators(data_location,
                                                    start=args.start,
                                                    end=args.end)))
    unknowns = []
    redetect_fun = partial(redetect_protocol, metadata)
    if args.parallel == 1:
        pool = Pool()
        for unk in tqdm(pool.imap(redetect_fun, protocols), total=len(protocols)):
            unknowns.extend(unk)
    else:
        for protocol in tqdm(protocols, total=len(protocols)):
            unk = redetect_fun(protocol)
            unknowns.extend(unk)

    unknowns = pd.DataFrame(unknowns, columns=['protocol_id', 'uuid']+["gender", "party", "other"])
    print('Proportion of metadata identified for unknowns:')
    print((unknowns[["gender", "party", "other"]] != '').sum() / len(unknowns))
    unknowns.drop_duplicates().to_csv(f"{args.processed_metadata_folder}/unknowns.csv", index=False)


    print("redetect seems to have finished successfully. Now run `cur-mot/split_into_sections --nextprev-only`.")
    # TODO: move abovementioned --nextprev-only function to pyriksdagen; import and run here




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)


    parser.add_argument("-s", "--start", type=int, default=1867, help="Start year")
    parser.add_argument("-e", "--end", type=int, default=2022, help="End year")
    parser.add_argument("-r", "--records-folder",
                        type=str,
                        default=None,
                        help="(optional) Path to records folder, defaults to environment var or `data/`")
    parser.add_argument("-m", "--metadata-root",
                        type=str,
                        default=None,
                        help="(optional) Path to metadata root folder, defaults to environment var or `data/`")
    parser.add_argument("-p", "--protocol",
                        type=str,
                        default=None,
                        help="operate on a single protocol")
    parser.add_argument("--parallel", type=int, default=1, help="N parallel processes (default=1)")
    parser.add_argument("--processed-metadata-folder", type=str, default="input/matching")
    args = parser.parse_args()
    main(args)
