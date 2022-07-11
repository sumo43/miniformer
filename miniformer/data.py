import zipfile
import urllib.request as request
import os

# english to spanish dataset
DATA_URL = 'https://www.manythings.org/anki/spa-eng.zip'
DATA_DIR = 'data'

def load_data(data_loc, toy=False):
    # need to have location of spa.txt
    # this dataset has multiple definitions for some words, we just skip them
    data=[]
    with open(data_loc, 'r') as f:
        for line in f:
            data.append(line)
    
    if toy:
        data = data[:300000]

    return data