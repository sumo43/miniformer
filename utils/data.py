import zipfile
import urllib.request as request
import os

# english to spanish dataset
DATA_URL = 'https://www.manythings.org/anki/spa-eng.zip'
DATA_DIR = 'data'

def load_data(mp, data_loc=os.path.join(os.getcwd(), DATA_DIR, 'spa.txt')):
    # need to have location of spa.txt
    # this dataset has multiple definitions for some words, we just skip them

    with open(data_loc, 'r') as f:
        i=0
        train = []
        labels = []

        for line in f:
            i+=1
            sentence_en, sentence_es = line.split('\t')[:2]

            train.append(sentence_en)
            labels.append(sentence_es)

        mp.set_ds_size(i) 
    
    return train, labels