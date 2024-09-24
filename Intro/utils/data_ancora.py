import pandas as pd
import numpy as np

from conllu import parse_incr
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from gensim.models import KeyedVectors

# enable tqdm in pandas - progress_map doesnt exist without
tqdm.pandas()

from utils.constants import ancora_train_file, ancora_embeddings_file, ancora_val_file, ancora_test_file


def read_tags(filename):
    # function reads the CoNLL-U file corresponding to a dataset and returns a pandas dataframe that
    # combines all tokens in a sentence into a single row with two columns, one
    # for the words, and one for the POS tags in the corresponding sentence:
    data = {'words': [], 'tags': []}
    with open(filename, 'r', encoding='utf-8') as f:
        for sent in parse_incr(f):
            words = [tok['form'] for tok in sent]
            tags = [tok['upos'] for tok in sent]

            data['words'].append(words)
            data['tags'].append(tags)
    return pd.DataFrame(data)

def load_embeddings():
    glove = KeyedVectors.load_word2vec_format(ancora_embeddings_file)
    print(glove.vectors.shape)

    # these embeddings already include <unk>
    unk_tok = '<unk>'
    unk_id = glove.key_to_index[unk_tok]

    # add padding embedding
    pad_tok = '<pad>'
    pad_emb = np.zeros(300)
    glove.add_vector(pad_tok, pad_emb)
    pad_tok_id = glove.key_to_index[pad_tok]

    return glove, unk_id, pad_tok_id


def preprocess(words):
    # Preprocessing: lower case all words, and replace all numbers with '0'
    result = []
    for w in words:
        w = w.lower()
        if w.isdecimal():
            w = '0'
        result.append(w)
    return result

def get_ids(tokens, key_to_index, unk_id=None):
    return [key_to_index.get(tok, unk_id) for tok in tokens]


def get_word_ids(tokens, glove, unk_id):
    return get_ids(tokens, glove.key_to_index, unk_id)


def get_tag_ids(tags, tag_to_index):
    return get_ids(tags, tag_to_index)


def create_vocab_of_pos_tags(train_df):
    # construct a vocabulary of POS tags. Once again, we generate a
    # list of tags using explode(), which linearizes our sequence of sequence
    # of tags, and remove repeated tags using unique(). We also add a special
    # tag for the padding token:

    pad_tag = '<pad>'
    index_to_tag = train_df['tags'].explode().unique().tolist() + [pad_tag]
    tag_to_index = {t: i for i, t in enumerate(index_to_tag)}
    pad_tag_id = tag_to_index[pad_tag]

    return tag_to_index, index_to_tag, pad_tag_id


def create_dataframe(file, glove, unk_id, tag_to_index=None, index_to_tag=None, pad_tag_id=None):
    df = read_tags(file)
    df['words'] = df['words'].progress_map(preprocess)
    print(df.head())

    # add a new column to the dataframe that stores the word ids
    # corresponding to the embedding vocabulary.
    df['word ids'] = df['words'].progress_map(
        lambda x: get_word_ids(x, glove, unk_id)
    )

    # This is only for training
    if tag_to_index is None:
        tag_to_index, index_to_tag, pad_tag_id = create_vocab_of_pos_tags(df)

    df['tag ids'] = df['tags'].progress_map(
        lambda x: get_tag_ids(x, tag_to_index)
    )

    return df, tag_to_index, index_to_tag, pad_tag_id


def get_split_dataframes():
    glove, unk_id, pad_tok_id = load_embeddings()
    train_df, tag_to_index, index_to_tag, pad_tag_id = create_dataframe(ancora_train_file, glove, unk_id, tag_to_index=None, index_to_tag=None, pad_tag_id=None)
    val_df, _, _, _ = create_dataframe(ancora_val_file, glove, unk_id, tag_to_index=tag_to_index)
    test_df, _, _, _ = create_dataframe(ancora_test_file, glove, unk_id, tag_to_index=tag_to_index)

    return glove, train_df, val_df, test_df, index_to_tag, pad_tok_id, pad_tag_id


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = torch.tensor(self.x[index])
        y = torch.tensor(self.y[index])
        return x, y



# HUGGING FACE
def get_dataset_dict():
    train_df = read_tags(ancora_train_file)
    valid_df = read_tags(ancora_val_file)
    test_df  = read_tags(ancora_test_file)

    tags = train_df['tags'].explode().unique()
    index_to_tag = {i: t for i, t in enumerate(tags)}
    tag_to_index = {t: i for i, t in enumerate(tags)}

    from datasets import Dataset, DatasetDict
    ds = DatasetDict()
    ds['train'] = Dataset.from_pandas(train_df)
    ds['validation'] = Dataset.from_pandas(valid_df)
    ds['test'] = Dataset.from_pandas(test_df)

    return ds, tags, index_to_tag, tag_to_index



if __name__ == '__main__':
    import os
    os.chdir("../")

    train_df, val_df, test_df = get_split_dataframes()

