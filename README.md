# Scripts – Data curation and processing logic for the Swedish Parliament Corpus 

This contains various scripts for curating and working with data from Swedish Riksdag. This repo is "internal" in some sense -- we make no effort to maintain compatibility or to provide really thorough documentation, and this repo is not intended as part of the project's API. Nevertheless, we feel that users might find some utility in these example scripts.

## General setup and use

The general recommendation is to set up a python virtual environment for working with this data set and these scripts. Do that how you like -- below is just one example of how it can be done. We're working with Python 3.8 due to compatibility issues with e.g. tensor flow.

### Setting up an environment

Set up a conda environment : Follow the steps [here](https://www.tensorflow.org/install/pip).

With the environment active, install the pyriksdagen module, either from PyPi

```
pip install pyriksdagen
```

or from a local copy in the [pyriksdagen repo](https://github.com/swerik-project/pyriksdagen)

```
pip install .
```

### The LazyArchive

The `LazyArchive()` class attempts to connect to the KB labs in the lazyest way possible. If you'll use the scripts often, it's worthwhile to set 3 environment variables:

	KBLMYLAB=https://betalab.kb.se
	KBLUSER=
	KBLPASS=

We are phasing out reliance on kblabb servers, and this will soon be deprecated.

They can be added to the environment variables, e.g. `~/miniconda3/envs/tf/etc/conda/activate.d/env_vars.sh`. If these are not present, you will be prompted for the username and password.


## Curating Records data


Most scripts take `--start` YEAR and `--end` YEAR arguments to define a span of time to operate on. Other options are noted in with the file below.

-1. Create new curation branch from dev.

	git checkout -b curation-<decade_start_year>s dev

1. Generate an input csv by querying protocol packages using `scripts/query2csv.py`
	- this creates `input/protocols/scanned.csv` or `input/protocols/digital_originals.csv`, to be read by `scripts/pipeline.py`
	- with the `-m` option the script will create year directories in `corpus/protocols/` if they don't already exist
    - ~~obs., unlike the other scripts use of `--start` and `--end` to define a range of dates is *exclusive* of the end year~~
        – updated to behave like the other scripts
    – obs. 2, a potential problem is that this doesn't handle the two-year formats - 199495

2. Compile parlaclarin for years queried in (1) with `scripts/pipeline.py`
    – make sure `input/raw/` exists.

3. Look for introductions with `scripts/classify_intros.py`
	- this creates `input/segmentation/intros.csv`
    - had to add `/home/bob/miniconda3/envs/tf/lib/python3.9/site-packages/nvidia/cublas/lib/` to $LD_LIBRARY_PATH

4. Run `scripts/resegment.py` to segment and label introductions in `corpus/protocols/<year>/*.xml` files

5. Run `scripts/add_uuid.py` to make sure any new segments have a uuid.

6. Run `scripts/find_dates.py` to find marginal notes with dates and add dates to metadata.

7. Run `scripts/build_classifier.py` (the classifier doesn't need to be built every time)
	different args!?
	- `--datapath` : needs a file currently at `input/curation/classifier_data.csv` (but how is this file generated? it's a mystery... it just exists)
	- `--epochs` (can use the default)
	- writes to the `segment-classifier/` ... how does it relate to years of protocols? it doesn't – it's apparently trained generally and `scripts/reclassify.py` allows to specify which years are operated on

8. Run `scripts/reclassify.py` to reclassify utterances and notes
    - nb. `build_classifier` writes to `segment-classifier/`, but this reads from `input/segment-classifier/`, so the output needs to be moved, or we can fix the discrepancy
    - do this one year at a time for dolan's sakie
        - `for year in {START..END}; do python scripts/reclassify.py -s $year -e $year; done`

9. Run `add_uuid.py` again.

10. Run `scripts/dollar_sign_replace.py` to replace dollar signs.

11. Run `scripts/fix_capitalized_dashes.py`.

12. Run `scripts/wikidata_process.py` (makes metadata available for redetect.py)

13. Run `scripts/redetect.py`.

14. Run `scripts/split_into_sections.py`.


## Quality Control

1. generate a sample for by decade with `sample_pages_new.py`. 
	- This generates a csv file in `input/quality_control/sample_<decade-start-year>.csv` and a list of protocols in the sample `input/quality_control/sample_<decade-start-year>.txt`
	
2. Add (`git-add_QC-sample.sh` for the lazy) and commit the sample to working branch.

3. Populate the quality control csv file with `populate-QC-sample-test.py`
	- sample protocols need to be on the local machine where the script is run. Since it pops open protocols in github an originals in betalab in a browser, this script doesn't play nice with working over ssh
	- QC should distinguish between the same segment classes that `scripts/reclassify.py` produces <u> and <note>. Other classes may become relevant later.

4. Does data pass QC test? If yes, add and push the rest of the protocols.

