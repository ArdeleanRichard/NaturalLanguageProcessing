import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def find_label_count(df):
    # Checking the distribution
    label_counts = df['spam'].value_counts()
    print("Label distribution:\n", label_counts)

def bar_plot(df):
    # Barplot for spam label distribution
    sns.countplot(x='spam', data=df)
    plt.title('Spam vs Non-Spam Emails')
    plt.show()


def distribution_plot_of_length(df):
    # Email length distribution (number of words in the text)
    df['email_length'] = df['text'].apply(lambda x: len(x.split()))

    # Mean and Standard Deviation of email lengths
    mean_length = np.mean(df['email_length'])
    std_length = np.std(df['email_length'])
    print("Data Statistics")
    print(f"-> Mean email length: {mean_length:.2f} words")
    print(f"-> StdDev of email length: {std_length:.2f} words")

    # Distribution plot of email lengths
    plt.figure(figsize=(10, 6))
    sns.displot(df['email_length'], kde=True)
    plt.title('Distribution of Email Lengths')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def explore_length(df):
    # Exploring short and long emails
    short_email = df[df['email_length'] <= 50]['text'].iloc[0]  # Example of short email
    long_email = df[df['email_length'] > 1000]['text'].iloc[0]  # Example of long email

    print("Data Exploration")
    print("-> Short email (<=50 words):\n--->", short_email)
    print("-> Long email (>1000 words):\n--->", long_email)


def correlation_analysis(df):
    # Add a new column for length
    df['length'] = df['text'].apply(len)

    # Correlation analysis
    correlation = df['length'].corr(df['spam'])
    print(f'Correlation between email length and spam label: {correlation:.4f}')

    # You can visualize it as well
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='spam', y='length', data=df)
    plt.title('Email Length by Spam Label')
    plt.xlabel('Spam')
    plt.ylabel('Email Length')
    plt.show()



if __name__ == '__main__':
    df = pd.read_csv('../data/spam/emails.csv')
    find_label_count(df)
    bar_plot(df)
    explore_length(df)
    distribution_plot_of_length(df)
    correlation_analysis(df)