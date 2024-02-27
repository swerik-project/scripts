#!/usr/bin/env python3
from tqdm import tqdm
import pandas as pd


def main(args):
	person = pd.read_csv(f"{args.metadata_folder}/person.csv")
	catalog = pd.read_csv(f"{args.qa_folder}/known_mps/catalog.csv", sep=';')
	DOBs = []
	for i, r in tqdm(catalog.iterrows(), total=len(catalog)):
		person_dobs = person[person['wiki_id'] == r['wiki_id']]
		if person_dobs.empty:
			DOBs.append(None)
		elif len(person_dobs) > 1:
			DOBs.append("Multival")
		else:
			DOBs.append(person_dobs.iloc[0]['born'])

	catalog['born'] = DOBs

	catalog.to_csv(args.outpath, sep=';')

if __name__ == '__main__':
	import argparse
	argparser = argparse.ArgumentParser(description=__doc__)
	argparser.add_argument("--metadata_folder", type=str, default="corpus/metadata")
	argparser.add_argument("--qa_folder", type=str, default="corpus/quality_assessment")
	argparser.add_argument("--outpath", type=str, default="corpus/quality_assessment/known_mps/catalog.csv")
	args = argparser.parse_args()
	main(args)
