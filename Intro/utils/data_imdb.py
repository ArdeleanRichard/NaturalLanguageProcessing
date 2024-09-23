from glob import glob

import torch

from utils.constants import imdb_train_neg_folder, imdb_train_pos_folder, imdb_test_pos_folder, imdb_test_neg_folder

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def get_data_files():
    pos_train_files = glob(imdb_train_pos_folder)
    neg_train_files = glob(imdb_train_neg_folder)
    pos_test_files = glob(imdb_test_pos_folder)
    neg_test_files = glob(imdb_test_neg_folder)

    print(f'Number of positive reviews: {len(pos_train_files)}')
    print(f'Number of negative reviews: {len(neg_train_files)}')

    return pos_train_files, neg_train_files, pos_test_files, neg_test_files

def get_data_count_vectorized():
    pos_train, neg_train, pos_test, neg_test = get_data_files()

    cv, doc_term_matrix = get_count_vectorized_data(pos_train, neg_train)

    X_train = doc_term_matrix.toarray()
    y_train = get_train_labels(pos_train, neg_train)

    X_test, y_test = get_test_data(cv, pos_test, neg_test)

    return X_train, y_train, X_test, y_test


def get_data_count_vectorized2():
    pos_train, neg_train, pos_test, neg_test = get_data_files()

    cv, doc_term_matrix = get_count_vectorized_data(pos_train, neg_train)

    X_train = doc_term_matrix.toarray()
    y_train = get_train_labels(pos_train, neg_train)

    X_test, y_test = get_test_data(cv, pos_test, neg_test)

    # Append 1s to the xs; this will allow us to multiply by the weights and
    # the bias in a single pass.
    # Make an array with a one for each row/data point
    X_train = np.column_stack((X_train, np.ones(X_train.shape[0])))
    X_test = np.column_stack((X_test, np.ones(X_test.shape[0])))

    return X_train, y_train, X_test, y_test


def get_data_count_vectorized_pytorch():
    pos_train, neg_train, pos_test, neg_test = get_data_files()

    cv, doc_term_matrix = get_count_vectorized_data(pos_train, neg_train)

    X_train = doc_term_matrix.toarray()
    y_train = get_train_labels(pos_train, neg_train)
    X_test, y_test = get_test_data(cv, pos_test, neg_test)


    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    return X_train, y_train, X_test, y_test


def get_count_vectorized_data(pos_train, neg_train):
    # Review 1: "I liked the movie. My friend liked it too."
    # Review 2: "I hated it. Would not recommend."
    #
    # {'would': 0,
    # 'hated': 1,
    # 'my': 2,
    # 'liked': 3,
    # 'not': 4,
    # 'it': 5,
    # 'movie': 6,
    # 'recommend': 7,
    # 'the': 8,
    # 'I': 9,
    # 'too': 10,
    # 'friend': 11}
    #
    # Using this mapping, we can encode the two reviews as follows:
    # Review 1: [0, 0, 1, 2, 0, 1, 1, 0, 1, 1, 1, 1]
    # Review 2: [1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0]


    # Now, we will use a CountVectorizer to read the text files, tokenize them,
    # acquire a vocabulary from the training data, and encode it in a document-term matrix
    # in which each row represents a review, and each column represents a term in the vocabulary.
    # Each element in the matrix represents the number of times term appears in example.
    cv = CountVectorizer(input='filename')
    doc_term_matrix = cv.fit_transform(pos_train + neg_train)

    print(f"Dictionary shape: {doc_term_matrix.shape}")
    # print(doc_term_matrix)

    return cv, doc_term_matrix


def get_train_labels(pos_train, neg_train):
    # training labels
    # We assign a label of one to positive reviews, and a label of zero to negative ones.
    # Note that the first half of the reviews are positive and the second half are negative
    y_pos = np.ones(len(pos_train))
    y_neg = np.zeros(len(neg_train))
    y_train = np.concatenate([y_pos, y_neg])
    print(y_train)

    return y_train


def get_test_data(cv, pos_test, neg_test):
    # testing data
    # Note that this time we use the transform() method of the CountVectorizer,
    # instead of the fit_transform() method that we used above.
    # This is because we want to use the learned vocabulary in the test set,
    # instead of learning a new one.
    doc_term_matrix = cv.transform(pos_test + neg_test)
    X_test = doc_term_matrix.toarray()
    y_pos = np.ones(len(pos_test))
    y_neg = np.zeros(len(neg_test))
    y_test = np.concatenate([y_pos, y_neg])

    return X_test, y_test