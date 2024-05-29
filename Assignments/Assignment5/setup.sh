#!/bin/bash

# create virtual env
python -m venv env
# activate env
source ./env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

ipython kernel install --user --name=env

deactivate