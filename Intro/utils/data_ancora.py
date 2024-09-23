import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from conllu import parse_incr
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from gensim.models import KeyedVectors

# enable tqdm in pandas - progress_map doesnt exist without
tqdm.pandas()

from utils.constants import ancora_train_file, ancora_embeddings_file
from collections import defaultdict


def read_tags(filename):
    data = {'words': [], 'tags': []}
    with open(filename, 'r', encoding='utf-8') as f:
        for sent in parse_incr(f):
            words = [tok['form'] for tok in sent]
            tags = [tok['upos'] for tok in sent]

            data['words'].append(words)
            data['tags'].append(tags)
    return pd.DataFrame(data)

if __name__ == '__main__':
    import os
    os.chdir("../")

    train_df = read_tags(ancora_train_file)
    print(train_df.head())

    glove = KeyedVectors.load_word2vec_format(ancora_embeddings_file)
    print(glove.vectors.shape)