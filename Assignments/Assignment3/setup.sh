#!/bin/bash

# create virtual env
python -m venv env
# activate env
source ./env/bin/activate

brew install python@3.8

pip install --upgrade pip
pip install -r requirements.txt


deactivate