"""
Find  introductions in the protocols using BERT. Used in tandem with resegment.py
"""
import pandas as pd
from lxml import etree
from transformers import AutoModelForSequenceClassification, BertTokenizerFast
from pyriksdagen.utils import protocol_iterators, elem_iter, get_data_location, TEI_NS
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
from pyriksdagen.dataset import IntroDataset
from functools import partial
import os
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)

def extract_elem_jointly(protocol, elem):
    text = elem.text.split()
    text = " ".join(text)
    u = elem.tag[-1] == "u"
    intro = elem.attrib.get("type") == "speaker"

    if intro:
        if elem.getnext().tag == f"{TEI_NS}u":
            print(f"concat intro ({text}) with next seg")
            text = text + " " + " ".join(elem.getnext()[0].text.split())
            print(f"result: {text}")

    return text, elem.get("{http://www.w3.org/XML/1998/namespace}id"), protocol

def extract_elem(protocol, elem):
    return elem.text, elem.get("{http://www.w3.org/XML/1998/namespace}id"), protocol


def extract_note_seg(protocol, heuristic=False):
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.parse(protocol, parser).getroot()
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
        print(df)
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
