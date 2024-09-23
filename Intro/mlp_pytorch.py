import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch import optim
from tqdm import tqdm

from utils.constants import labels, device
from utils.data_agnews import get_data_agnews


def create_model(X_train, y_train):
    # hyperparameters
    lr = 1.0
    n_epochs = 5
    n_examples = X_train.shape[0]
    n_feats = X_train.shape[1]
    n_classes = len(labels)

    # initialize the model, loss function, optimizer, and data-loader
    model = nn.Linear(n_feats, n_classes).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # train the model
    indices = np.arange(n_examples)
    for epoch in range(n_epochs):

        np.random.shuffle(indices)

        for i in tqdm(indices, desc=f'epoch {epoch + 1}'):
            # clear gradients
            model.zero_grad()

            # send datum to right device
            x = X_train[i].unsqueeze(0).to(device)
            y_true = y_train[i].unsqueeze(0).to(device)

            # predict label scores
            y_pred = model(x)

            # compute loss
            loss = loss_func(y_pred, y_true)

            # backpropagate
            loss.backward()

            # optimize model parameters
            optimizer.step()

    return model


def eval_model(model, X_test):
    # set model to evaluation mode
    model.eval()

    # don't store gradients
    with torch.no_grad():
        X_test = X_test.to(device)
        y_pred = torch.argmax(model(X_test), dim=1)
        y_pred = y_pred.cpu().numpy()
        print(classification_report(y_test, y_pred, target_names=labels))



if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_data_agnews()
    model = create_model(X_train, y_train)
    eval_model(model, X_test)

