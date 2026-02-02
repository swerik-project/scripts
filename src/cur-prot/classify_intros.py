"""
Find  introductions in the protocols using BERT. Used in tandem with resegment.py
"""

from functools import partial
import os
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
import pandas as pd
from pyriksdagen.dataset import IntroDataset
from pyriksdagen.io import parse_tei
from pyriksdagen.utils import (
    elem_iter,
    TEI_NS
)
from trainerlog import get_logger
from transformers import AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = get_logger(name="trainlog.predict_intro", level="DEBUG")

def extract_elem_jointly(protocol, elem):
    text = elem.text.split()
    text = " ".join(text)
    u = elem.tag[-1] == "u"
    intro = elem.attrib.get("type") == "speaker"

    if intro:
        next_elem = elem.getnext()
        if next_elem.tag == f"{TEI_NS}u":
            next_elem = next_elem[0]
            if next_elem.text is not None:
                logger.debug(f"concat intro ({text}) with next seg")
                u_text = " ".join(next_elem.text.split())
                #if "." in u_text:
                #    u_text = u_text.split(".")[0] + "."
                text = text + " " + u_text
                #print(f"result: {text}")

    return text, elem.get("{http://www.w3.org/XML/1998/namespace}id"), protocol

def extract_elem(protocol, elem):
    return elem.text, elem.get("{http://www.w3.org/XML/1998/namespace}id"), protocol


def extract_note_seg(protocol, heuristic=False):
    root, ns = parse_tei(protocol)
    data = []
    extract_elem_fun = extract_elem
    if heuristic:
        extract_elem_fun = extract_elem_jointly
    for tag, elem in elem_iter(root):
        if tag == 'note':
            data.append(extract_elem_fun(protocol, elem))
        elif tag == 'u':
            data.extend(list(map(partial(extract_elem_fun, protocol), elem)))
    return data


def predict_intro(df, cuda):
    model = AutoModelForSequenceClassification.from_pretrained("jesperjmb/parlaBERT")
    if cuda:
        model = model.to('cuda')
    test_dataset = IntroDataset(df)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4)

    intros = []
    with torch.no_grad():
        for texts, xml_ids, file_path in tqdm(test_loader, total=len(test_loader)):

            if cuda:
                output = model( input_ids=texts["input_ids"].squeeze(dim=1).to('cuda'),
                                token_type_ids=texts["token_type_ids"].squeeze(dim=1).to('cuda'),
                                attention_mask=texts["attention_mask"].squeeze(dim=1).to('cuda'))
            else:
                output = model( input_ids=texts["input_ids"].squeeze(dim=1),
                            token_type_ids=texts["token_type_ids"].squeeze(dim=1),
                            attention_mask=texts["attention_mask"].squeeze(dim=1))

            preds = torch.argmax(output[0], dim=1)
            intros.extend([[file_path, xml_id] for file_path, xml_id, pred in zip(file_path, xml_ids, preds) if pred == 1])
    return pd.DataFrame(intros, columns=['file_path', 'id'])



def main(args):
    intros = []
    protocols = args.records
    protocols = [os.path.split(p) for p in protocols]
    protocol_df = pd.DataFrame(protocols, columns=['folder', 'file'])
    protocol_df = protocol_df.sort_values(by=['folder', 'file'])
    folders = sorted(set(protocol_df['folder']))

    for folder in folders:
        files = protocol_df.loc[protocol_df['folder'] == folder, 'file'].tolist()
        data = []
        for file in tqdm(files, total=len(files)):
            data.extend(extract_note_seg(os.path.join(folder, file), heuristic=args.join_heuristic))
        df = pd.DataFrame(data, columns=['text', 'id', 'file_path'])
        logger.debug(f"{df}")
        N = len(df)
        null_data = df[df.isnull().any(axis=1)]
        df = df.dropna()
        N_prime = len(df)
        if N != N_prime:
            logger.warning(f"{N - N_prime} null rows were omitted.")
            logger.debug(null_data)
        df = predict_intro(df, cuda=args.cuda)
        intros.append(df)

    df = pd.concat(intros)
    df.to_csv(args.outpath, index=False)




if __name__ == "__main__":
    parser = fetch_parser("records")
    parser.add_argument("--cuda", action="store_true", help="Set this flag to run with cuda.")
    parser.add_argument("--join_heuristic", action="store_true", help="Jointly predict intros that have been previously split")
    parser.add_argument("--outpath", default="input/segmentation/intros.csv")
    args = impute_args(parser.parse_args())
    main(args)
