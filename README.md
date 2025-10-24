# Scripts – Data curation and processing logic for the Swedish Parliament Motion Corpus 

This repository contains the necessary scripts for the processing and curation of the
Swedish Parliament Motion corpus. The scripts are written in Python, and documented
with docstrings that can be seen by running each script with the ```--help``` flag.

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
