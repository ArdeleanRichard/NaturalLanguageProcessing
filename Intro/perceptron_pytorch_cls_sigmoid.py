import numpy as np
from tqdm import tqdm

from utils.validation import binary_classification_report
from utils.data import get_data_count_vectorized, get_data_count_vectorized2


def sigmoid(z):
    if -z > np.log(np.finfo(float).max):
        return 0.0
    return 1 / (1 + np.exp(-z))

def create_perceptron_model(X_train, y_train):
    n_examples, n_features = X_train.shape
    w = np.random.random(n_features)

    lr = 1e-1
    n_epochs = 10

    indices = np.arange(n_examples)
    for epoch in range(n_epochs):

        # randomize the order in which training examples are seen in this epoch
        np.random.shuffle(indices)

        # traverse the training data
        for i in tqdm(indices, desc=f'epoch {epoch+1}'):
            x = X_train[i]
            y = y_train[i]

            # calculate the derivative of the cost function for this batch
            deriv_cost = (sigmoid(x @ w) - y) * x

            # update the weights
            w = w - lr * deriv_cost

    return w


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_data_count_vectorized2()
    w = create_perceptron_model(X_train, y_train)

    # Using the model is easy: multiply the document-term matrix by the learned weights and add the bias.
    # We use Python's @ operator to perform the matrix-vector multiplication.
    y_pred = X_test @ w > 0
    performance_dict = binary_classification_report(y_test, y_pred)
    print(performance_dict)
