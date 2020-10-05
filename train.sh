#!/bin/bash

cd data/GraphConstruction
python constituency.py

cd ../../src
mkdir checkpoint_dir
mkdir checkpoint_dir/valid
mkdir output

# echo -----------pretrained embedding generating-----------
python pretrained_embedding.py -pretrained_embedding="*where your glove text saved*"
# echo ------------Begin training---------------------------
python graph2tree.py
# echo -----------------------------------------------------
python sample_valid.py 