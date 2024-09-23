import numpy as np
from gensim.models import KeyedVectors
from collections import Counter
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.constants import agnews_embeddings_file, agnews_train_file, agnews_test_file
from utils.data_agnews import get_split_data_with_tokens


def count_unknown_words(data, vocabulary):
    counter = Counter()
    for row in tqdm(data):
        counter.update(tok for tok in row if tok not in vocabulary)
    return counter


def analysis_embeddings(glove, train_df):
    # Our analysis indicates that only 1.25% of the tokens are not accounted
    # for in the embeddings vocabulary. Further, the most common unknown
    # words seem to be URL fragments.

    # find out how many times each unknown token occurrs in the corpus
    c = count_unknown_words(train_df['tokens'], glove.key_to_index)

    # find the total number of tokens in the corpus
    total_tokens = train_df['tokens'].map(len).sum()

    # find some statistics about occurrences of unknown tokens
    unk_tokens = sum(c.values())
    percent_unk = unk_tokens / total_tokens
    distinct_tokens = len(list(c))

    print(f'Total number of tokens: {total_tokens:,}')
    print(f'Number of unknown tokens: {unk_tokens:,}')
    print(f'Number of distinct unknown tokens: {distinct_tokens:,}')
    print(f'Percentage of unknown tokens: {percent_unk:.2%}')
    print('Top 10 unknown words:')
    for token, n in c.most_common(10):
        print(f'\t{n}\t{token}')


def handle_unknown(glove):
    # Our analysis indicates that only 1.25% of the tokens are not accounted
    # for in the embeddings vocabulary. Further, the most common unknown
    # words seem to be URL fragments. This is encouraging. However, for
    # more robustness, we will introduce a couple of special embeddings that
    # are often needed when dealing with word embeddings. The first one is an
    # embedding used to represent unknown words. A common strategy is to
    # use the average of all the embeddings in the vocabulary for this purpose.
    # The second embedding we will add will be used for padding. Padding is
    # required when we want to train with (mini-)batches because the lengths
    # of all the examples in a given batch have to match in order for the batch
    # to be efficiently processed in parallel. The padding embedding consists
    # only of zeros, which essentially excludes these virtual tokens from the
    # forward/backward passes. None of these embeddings are included in the
    # pretrained GloVe embeddings, but other pretrained embeddings may
    # already include them, so it is a good idea to check if they are included
    # with the embeddings we are using before adding them

    # string values corresponding to the new embeddings
    unk_tok = '[UNK]'
    pad_tok = '[PAD]'

    # initialize the new embedding values
    unk_emb = glove.vectors.mean(axis=0)
    pad_emb = np.zeros(300)

    # add new embeddings to glove
    glove.add_vectors([unk_tok, pad_tok], [unk_emb, pad_emb])

    # get token ids corresponding to the new embeddings
    unk_id = glove.key_to_index[unk_tok]
    pad_id = glove.key_to_index[pad_tok]

    # print(unk_id, pad_id)

    return unk_id, pad_id



def create_vocab(train_df):
    threshold = 10
    tokens = train_df['tokens'].explode().value_counts()
    vocabulary = set(tokens[tokens > threshold].index.tolist())
    print(f'vocabulary size: {len(vocabulary):,}')

    return vocabulary


def get_id(tok, glove, vocabulary, unk_id):
    # return unk_id for infrequent tokens too
    if tok in vocabulary:
        return glove.key_to_index.get(tok, unk_id)
    else:
        return unk_id


def token_ids(tokens, max_tokens, glove, vocabulary, unk_id, pad_id):
    # gets a list of tokens and returns a list of token ids,
    # with padding added accordingly
    tok_ids = [get_id(tok, glove, vocabulary, unk_id) for tok in tokens]
    pad_len = max_tokens - len(tok_ids)
    return tok_ids + [pad_id] * pad_len

def add_padded_sequences(df, glove, vocabulary, unk_id, pad_id):
    # find the length of the longest list of tokens
    max_tokens = df['tokens'].map(len).max()

    # add new column to the dataframe
    df['token ids'] = df['tokens'].progress_map(
        lambda x: token_ids(x, max_tokens, glove, vocabulary, unk_id, pad_id)
    )
    return df



def get_split_data_with_embeddings():
    train_df, val_df, test_df = get_split_data_with_tokens(agnews_train_file, agnews_test_file)

    # Load GloVe word embeddings:
    glove = KeyedVectors.load_word2vec_format(agnews_embeddings_file, no_header=True)

    # The word embeddings have been pretrained in a different corpus, so it would be a good idea
    # to estimate how good our tokenization matches the GloVe vocabulary.
    analysis_embeddings(glove, train_df)
    unk_id, pad_id = handle_unknown(glove)
    vocabulary = create_vocab(train_df)

    # add a new column to our dataframe that will contain the padded sequences of token ids
    train_df = add_padded_sequences(train_df, glove, vocabulary, unk_id, pad_id)
    val_df = add_padded_sequences(val_df, glove, vocabulary, unk_id, pad_id)
    test_df = add_padded_sequences(test_df, glove, vocabulary, unk_id, pad_id)

    return glove, train_df, val_df, test_df


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


if __name__ == '__main__':
    import os
    os.chdir("../")

    get_split_data_with_embeddings()


