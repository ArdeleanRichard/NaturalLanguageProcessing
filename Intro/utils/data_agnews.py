import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.constants import train_file, labels


def get_data():
    train_df = pd.read_csv(train_file, header=None, skiprows=1)
    train_df.columns = ['class index', 'title', 'description']

    classes = train_df['class index'].map(lambda i: labels[i - 1])
    train_df.insert(1, 'class', classes)
    # print(train_df.head())
    # pd.value_counts(train_df['class']).plot.bar()
    # plt.show()

    return train_df