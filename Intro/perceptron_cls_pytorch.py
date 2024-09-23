import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import numpy as np
from utils.data_imdb import get_data_count_vectorized, get_data_count_vectorized_pytorch
from utils.validation import binary_classification_report


def model_variation1(n_features):
    model = nn.Linear(n_features, 1)
    loss_func = nn.BCEWithLogitsLoss()

    return model, loss_func

def model_variation2(n_features):
    model = nn.Sequential(
        nn.Linear(n_features, 2),
        nn.LogSoftmax(dim=1),
    )
    loss_func = nn.NLLLoss()

    return model, loss_func

def create_perceptron_model(X_train, y_train):
    n_examples, n_features = X_train.shape

    lr = 1e-1
    n_epochs = 10

    model, loss_func = model_variation1(n_features)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    indices = np.arange(n_examples)
    for epoch in range(n_epochs):
        # randomize training examples
        np.random.shuffle(indices)

        # for each training example
        for i in tqdm(indices, desc=f'epoch {epoch + 1}'):
            x = X_train[i]  # for variation1
            y_true = y_train[i]  # for variation1
            # x = X_train[i].unsqueeze(0)  # for variation2
            # y_true = y_train[i].unsqueeze(0)  # for variation2

            # make predictions
            y_pred = model(x)

            # calculate loss
            loss = loss_func(y_pred[0], y_true) # for variation1
            # loss = loss_func(y_pred, y_true) # for variation2

            # calculate gradients through back-propagation
            loss.backward()

            # optimize model parameters
            optimizer.step()

            # ensure gradients are set to zero
            model.zero_grad()

    return model


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_data_count_vectorized_pytorch()
    model = create_perceptron_model(X_train, y_train)

    y_pred = model(X_test) > 0
    performance_dict = binary_classification_report(y_test, y_pred)
    print(performance_dict)
