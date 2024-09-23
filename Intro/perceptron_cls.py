import numpy as np
from tqdm import tqdm

from utils.data_imdb import get_data_count_vectorized
from utils.validation import binary_classification_report


def create_perceptron_model(X_train, y_train):
    n_examples, n_features = X_train.shape

    # Now we will initialize our model, in the form of an array of weights w of the same size
    # as the number of features in our dataset (i.e., the number of words in the vocabulary
    # acquired by CountVectorizer), and a bias term b. Both are initialized to zeros.
    w = np.zeros(n_features)
    b = 0

    # Now we will use the perceptron learning algorithm to learn the values of w and b
    # from our training data.
    n_epochs = 10

    indices = np.arange(n_examples)
    for epoch in range(n_epochs):
        n_errors = 0
        # randomize the order in which training examples are seen in this epoch
        np.random.shuffle(indices)
        # traverse the training data
        for i in tqdm(indices, desc=f'epoch {epoch + 1}'):
            x = X_train[i]
            y_true = y_train[i]

            # the perceptron decision based on the current model
            score = x @ w + b
            y_pred = 1 if score > 0 else 0

            if y_true == y_pred:
                continue
            # update the model is the prediction was incorrect
            elif y_true == 1 and y_pred == 0:
                w = w + x
                b = b + 1
                n_errors += 1
            elif y_true == 0 and y_pred == 1:
                w = w - x
                b = b - 1
                n_errors += 1
        if n_errors == 0:
            break

    return w, b



if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_data_count_vectorized()
    w, b = create_perceptron_model(X_train, y_train)

    # Using the model is easy: multiply the document-term matrix by the learned weights and add the bias.
    # We use Python's @ operator to perform the matrix-vector multiplication.
    y_pred = (X_test @ w + b) > 0
    performance_dict = binary_classification_report(y_test, y_pred)
    print(performance_dict)
