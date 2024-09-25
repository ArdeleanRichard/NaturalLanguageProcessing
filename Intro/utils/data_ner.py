import pandas as pd
from sklearn.model_selection import train_test_split

from utils.constants import ner_file
from utils.data_ancora import load_embeddings, preprocess


def convert_data_to_sentences(df):
    # Fill empty "Sentence #" values with the previous one (forward fill)
    df['Sentence #'] = df['Sentence #'].replace('', None).ffill()
    df['Word'] = df['Word'].astype(str)

    # Remove the "Sentence: " part from the 'Sentence #' column to keep just the number
    df['Sentence #'] = df['Sentence #'].str.replace('Sentence: ', '').astype(int)

    # Group the dataframe by 'Sentence #' and create lists of 'Word', 'POS', and 'Tag'
    grouped_df = df.groupby('Sentence #').agg({
        'Word': lambda x: list(x),
        'POS': lambda x: list(x),
        'Tag': lambda x: list(x)
    }).reset_index()

    print(grouped_df["Word"])

    grouped_df.columns = ['Sentence #', 'words', 'postags', 'nertags']

    return grouped_df


def create_vocab_of_pos_tags(df):
    # construct a vocabulary of POS tags. Once again, we generate a
    # list of tags using explode(), which linearizes our sequence of sequence
    # of tags, and remove repeated tags using unique(). We also add a special
    # tag for the padding token:

    pad_tag = '<pad>'
    index_to_tag = df['postags'].explode().unique().tolist() + [pad_tag]
    tag_to_index = {t: i for i, t in enumerate(index_to_tag)}
    pad_tag_id = tag_to_index[pad_tag]

    return tag_to_index, index_to_tag, pad_tag_id



def create_vocab_of_ner_tags(df):
    # construct a vocabulary of POS tags. Once again, we generate a
    # list of tags using explode(), which linearizes our sequence of sequence
    # of tags, and remove repeated tags using unique(). We also add a special
    # tag for the padding token:

    pad_tag = '<pad>'
    index_to_tag = df['nertags'].explode().unique().tolist() + [pad_tag]
    tag_to_index = {t: i for i, t in enumerate(index_to_tag)}
    pad_tag_id = tag_to_index[pad_tag]

    return tag_to_index, index_to_tag, pad_tag_id

def get_ids(tokens, key_to_index, unk_id=None):
    return [key_to_index.get(tok, unk_id) for tok in tokens]


def get_word_ids(tokens, glove, unk_id):
    return get_ids(tokens, glove.key_to_index, unk_id)


def get_tag_ids(tags, tag_to_index):
    return get_ids(tags, tag_to_index)



def prepare_dataframe(df, glove, unk_id, tag_to_index=None, index_to_tag=None, pad_tag_id=None):
    df['words'] = df['words'].progress_map(preprocess)
    print(df.head())

    # add a new column to the dataframe that stores the word ids
    # corresponding to the embedding vocabulary.
    df['word ids'] = df['words'].progress_map(
        lambda x: get_word_ids(x, glove, unk_id)
    )

    # This is only for training
    # if tag_to_index is None:
    #     tag_to_index, index_to_tag, pad_tag_id = create_vocab_of_pos_tags(df)
    #
    # df['postag ids'] = df['postags'].progress_map(
    #     lambda x: get_tag_ids(x, tag_to_index)
    # )

    # This is only for training
    if tag_to_index is None:
        tag_to_index, index_to_tag, pad_tag_id = create_vocab_of_ner_tags(df)

    df['nertag ids'] = df['nertags'].progress_map(
        lambda x: get_tag_ids(x, tag_to_index)
    )

    return df, tag_to_index, index_to_tag, pad_tag_id


def get_split_dataframes(filename):
    df = pd.read_csv(filename, encoding='Windows-1252')
    df = convert_data_to_sentences(df)

    train_df, val_df = train_test_split(df, train_size=0.6)
    train_df.reset_index(inplace=True)
    val_df.reset_index(inplace=True)

    val_df, test_df = train_test_split(val_df, train_size=0.5)
    val_df.reset_index(inplace=True)
    test_df.reset_index(inplace=True)

    # glove, unk_id, pad_tok_id = load_embeddings()
    # train_df, tag_to_index, index_to_tag, pad_tag_id = prepare_dataframe(train_df, glove, unk_id, tag_to_index=None, index_to_tag=None, pad_tag_id=None)
    # val_df, _, _, _ = prepare_dataframe(val_df, glove, unk_id, tag_to_index=tag_to_index)
    # test_df, _, _, _ = prepare_dataframe(test_df, glove, unk_id, tag_to_index=tag_to_index)

    # return glove, train_df, val_df, test_df, tag_to_index, index_to_tag, pad_tok_id, pad_tag_id
    return train_df, val_df, test_df


if __name__ == '__main__':
    import os
    os.chdir("../")

    glove, train_df, val_df, test_df, tag_to_index, index_to_tag, pad_tok_id, pad_tag_id = get_split_dataframes(ner_file)
    print("Words and IDS:")
    print(train_df["words"].head())
    print(train_df["word ids"].head())

    # print("POS tags and POS ids")
    # print(train_df["postags"].head())
    # print(train_df["postag ids"].head())

    print("NER tags and NER ids")
    print(train_df["nertags"].head())
    print(train_df["nertag ids"].head())