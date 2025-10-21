#!/bin/bash

year=$1

python scripts/src/cur-mot/alto-pipeline.py -y $year
python scripts/src/cur-mot/add-uuid -y $year
python scripts/src/cur/mot/mv-fort.py -y $year
python scripts/src/cur/mot/mv-reg.py -y $year

# if fk or ak
python scripts/src/cur-mot/classify-fw.py -y $year
python scripts/src/cur-mot/list-absurd-ranges-per-page.py --list -y $year
#+ intervene here
python scripts/src/cur-mot/list-absurd-ranges-per-page.py --fix -y $year
python scripts/src/cur-mot/split-multimot-pages.py -y $year
# fi
