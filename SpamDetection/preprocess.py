import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # print("NLTK Stopwords:", stop_words)

    # Lowercase the text
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Join tokens back into a single string
    cleaned_text = ' '.join(cleaned_tokens)

    return cleaned_text


def data_split_by_label(df):
    # Splitting the data into training and test sets (80/20) with stratification on 'spam' labels
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['spam'], random_state=42)

    # Checking the distribution of the labels in train/test datasets
    print(f"Training label distribution:\n{train_df['spam'].value_counts(normalize=True)}")
    print(f"Test label distribution:\n{test_df['spam'].value_counts(normalize=True)}")

    return train_df, test_df

def data_split_by_label_and_length(df):
    # Create the 'email_length' column which contains the number of words in each email
    df['email_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))

    # Create bins for email lengths (short, medium, long)
    df['length_bin'] = pd.cut(df['email_length'], bins=[0, 100, 500, float('Inf')], labels=['short', 'medium', 'long'])

    # Splitting the data with stratification on both 'spam' and 'length_bin'
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[['spam', 'length_bin']], random_state=42)

    # Checking the distribution of spam and length_bin in train/test datasets
    print("Split Distribution")
    print(train_df.groupby(['spam', 'length_bin']).size())

    return train_df, test_df

def tfidf(df):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df)

    # Get the feature names (i.e., the words in the vocabulary)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    print(f"Feature Names: {feature_names}")

    # TF-IDF matrix (rows = documents, columns = words)
    dense_tfidf_matrix = tfidf_matrix.todense()
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

    # Display the TF-IDF matrix for a sample of documents
    print(pd.DataFrame(dense_tfidf_matrix[:5], columns=feature_names).head())


    # Hyperparameter Tuning: You can tweak parameters like
    #   max_features (limiting the vocabulary size),
    #   min_df (min document frequency), and
    #   ngram_range
    # to improve the model’s performance based on your specific dataset.

    return tfidf_vectorizer, np.array(dense_tfidf_matrix)


if __name__ == '__main__':
    df = pd.read_csv('../data/spam/emails.csv')
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    print(df[['text', 'cleaned_text']].head())