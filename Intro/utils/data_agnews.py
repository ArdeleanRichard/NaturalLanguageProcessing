import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

# enable tqdm in pandas - progress_map doesnt exist without
tqdm.pandas()

from utils.constants import train_file, labels, column_names, test_file
from collections import defaultdict

def get_data_from_file(file, train_data=False):
    df = pd.read_csv(file, header=None, skiprows=1)
    df.columns = column_names

    if train_data:
        classes = df['class index'].map(lambda i: labels[i - 1])
        df.insert(1, 'class', classes)

    # print(train_df.head())
    # pd.value_counts(train_df['class']).plot.bar()
    # plt.show()

    # clean data
    # title + description
    # Oil and Economy Cloud Stocks' Outlook (Reuters)  +    Reuters - Soaring crude prices plus worries\ab..
    title = df['title'].str.lower()
    descr = df['description'].str.lower()
    text = title + " " + descr
    df['text'] = text.str.replace('\\', ' ', regex=False)

    return df


def add_tokens_to_data(df):
    # [oil, and, economy, cloud, stocks, ', outlook, ... ]
    df['tokens'] = df['text'].progress_map(word_tokenize)

def create_vocab(train_df):
    # Here, we only keep the words that occur at least 10 times,
    # decreasing the memory needed and reducing the likelihood
    # that our vocabulary contains noisy tokens.
    threshold = 10
    # In order to create the vocabulary, we will need
    # to convert the Series of lists of tokens into a Series of tokens using
    # the explode() Pandas method. Then we will use the value_counts()
    # method to create a Series object in which the index are the tokens
    # and the values are the number of times they appear in the corpus.
    tokens = train_df['tokens'].explode().value_counts()
    tokens = tokens[tokens > threshold]
    id_to_token = ['[UNK]'] + tokens.index.tolist()
    token_to_id = {w: i for i, w in enumerate(id_to_token)}
    vocabulary_size = len(id_to_token)

    return token_to_id, vocabulary_size

def make_feature_vector(tokens, token_to_id, unk_id=0):
    vector = defaultdict(int)
    for t in tokens:
        i = token_to_id.get(t, unk_id)
        vector[i] += 1
    return vector

def add_features_to_data(train_df, token_to_id):

    # Using this vocabulary, we construct a feature vector for each news
    # article in the corpus. This feature vector will be encoded as a dictionary,
    # with keys corresponding to token ids, and values corresponding to the
    # number of times the token appears in the article. As above, the feature
    # vectors will be stored as a new column in the dataframe.

    # transform
    # [oil, and, economy, cloud, stocks, ', outlook, ... ]
    # into
    # {66: 1, 9: 2, 351: 2, 4565: 1, 158: 1, 116: 1
    train_df['features'] = train_df['tokens'].progress_map(
        lambda x: make_feature_vector(x, token_to_id)
    )

def make_dense(feats, vocabulary_size):
    x = np.zeros(vocabulary_size)
    for k, v in feats.items():
        x[k] = v
    return x

def get_tensor_data(df, vocabulary_size):
    # The final preprocessing step is converting the features and the class
    # indices into PyTorch tensors. Recall that we need to subtract one from
    # the class indices to make them zero-based.
    X = np.stack(df['features'].progress_map(
        lambda x: make_dense(x, vocabulary_size)
    ))
    y = df['class index'].to_numpy() - 1
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y)

    return X, y


def get_train_data(train_file):
    train_df = get_data_from_file(train_file, train_data=True)

    add_tokens_to_data(train_df)

    token_to_id, vocabulary_size = create_vocab(train_df)
    add_features_to_data(train_df, token_to_id)

    X_train, y_train = get_tensor_data(train_df, vocabulary_size)
    return X_train, y_train, token_to_id, vocabulary_size


def get_test_data(test_file, token_to_id, vocabulary_size):
    # repeat all preprocessing done above, this time on the test set
    test_df = get_data_from_file(test_file, train_data=False)

    add_tokens_to_data(test_df)
    add_features_to_data(test_df, token_to_id)

    X_test, y_test = get_tensor_data(test_df, vocabulary_size)

    return X_test, y_test



def get_data_agnews():
    X_train, y_train, token_to_id, vocabulary_size = get_train_data(train_file)
    X_test, y_test = get_test_data(test_file, token_to_id, vocabulary_size)
    return X_train, y_train, X_test, y_test



def get_split_data(train_file, test_file):
    train_df = get_data_from_file(train_file, train_data=True)
    test_df = get_data_from_file(test_file, train_data=False)

    add_tokens_to_data(train_df)
    add_tokens_to_data(test_df)

    train_df, val_df = train_test_split(train_df, train_size=0.8)
    train_df.reset_index(inplace=True)
    val_df.reset_index(inplace=True)

    print(f'train rows: {len(train_df.index):,}')
    print(f'dev rows: {len(val_df.index):,}')
    print(f'test rows: {len(test_df.index):,}')

    token_to_id, vocabulary_size = create_vocab(train_df)
    add_features_to_data(train_df, token_to_id)
    add_features_to_data(val_df, token_to_id)
    add_features_to_data(test_df, token_to_id)

    return train_df, val_df, test_df, vocabulary_size




class MyDataset(Dataset):
    def __init__(self, x, y, vocabulary_size):
        self.x = x
        self.y = y
        self.vocabulary_size = vocabulary_size

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = torch.zeros(self.vocabulary_size, dtype=torch.float32)
        y = torch.tensor(self.y[index])
        for k, v in self.x[index].items():
            x[k] = v
        return x, y


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_data_agnews()
    print(X_train.shape)

