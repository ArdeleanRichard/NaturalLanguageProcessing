import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess import preprocess_text, data_split_by_label, data_split_by_label_and_length, tfidf
from validation import get_performance, plot_confusion_matrix, is_spam
from visualization import find_label_count, bar_plot, explore_length, distribution_plot_of_length


if __name__ == '__main__':
    # Load data
    df = pd.read_csv('../data/spam/emails.csv')

    # Visualize data
    bar_plot(df)
    find_label_count(df)
    distribution_plot_of_length(df)
    explore_length(df)

    # Clean data
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    print(df[['text', 'cleaned_text']].head())

    # Split (train, test) data
    # train_df, test_df = data_split_by_label(df)
    train_df, test_df = data_split_by_label_and_length(df)

    # Transform into TF-IDF
    vectorizer, train_tfidf = tfidf(train_df['cleaned_text'])
    test_tfidf = vectorizer.transform(test_df['cleaned_text'])
    test_tfidf = np.array(test_tfidf.todense())


    # Train LogReg on data
    logreg = LogisticRegression()
    logreg.fit(train_tfidf, train_df['spam'])

    y_pred = logreg.predict(test_tfidf)

    # Performance
    get_performance(test_df['spam'], y_pred)
    plot_confusion_matrix(test_df['spam'], y_pred)

    # Test prediction
    test_text = "Congratulations! You've won a free ticket!"
    print(is_spam(vectorizer, logreg, test_text, clean=True))
    print(is_spam(vectorizer, logreg, test_text, clean=False))
    test_text2 = "Felicidades! Has ganado un ticket gratis!"
    print(is_spam(vectorizer, logreg, test_text, clean=True))
