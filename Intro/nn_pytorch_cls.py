import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix

from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.constants import agnews_labels, device, agnews_train_file, agnews_test_file
from utils.data_agnews import MyDataset, get_split_data_with_features
from utils.model_aux import plot_train_history


def prepare_data(hyperparam, train_df, val_df, test_df):
    train_ds = MyDataset(
        x=train_df['features'],
        y=train_df['class index'] - 1,
        vocabulary_size=hyperparam['input_dim'],
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=hyperparam["batch_size"],
        shuffle=hyperparam["shuffle"]
    )


    val_ds = MyDataset(
        x=val_df['features'],
        y=val_df['class index'] - 1,
        vocabulary_size=hyperparam['input_dim'],
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=hyperparam["batch_size"],
        shuffle=hyperparam["shuffle"])


    test_ds = MyDataset(
        x=test_df['features'],
        y=test_df['class index'] - 1,
        vocabulary_size=hyperparam['input_dim'],
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=hyperparam["batch_size"]
    )

    return train_dl, val_dl, test_dl

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)

def train_model(hyperparam, train_dl, val_dl):
    # initialize the model, loss function, optimizer, and data-loader
    model = Model(
        hyperparam["input_dim"],
        hyperparam["hidden_dim"],
        hyperparam["output_dim"],
        hyperparam["dropout"]
    ).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparam["lr"],
        weight_decay=hyperparam["weight_decay"])

    # lists used to store plotting data
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    # train the model
    for epoch in range(hyperparam["n_epochs"]):
        losses, acc = [], []

        # set model to training mode
        model.train()

        for X, y_true in tqdm(train_dl, desc=f'[train] epoch {epoch + 1} '):

            # clear gradients
            model.zero_grad()

            # send batch to right device
            X = X.to(device)
            y_true = y_true.to(device)

            # predict label scores
            y_pred = model(X)

            # compute loss
            loss = loss_func(y_pred, y_true)

            # compute accuracy
            gold = y_true.detach().cpu().numpy()
            pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)

            # accumulate for plotting
            losses.append(loss.detach().cpu().item())
            acc.append(accuracy_score(gold, pred))

            # backpropagate
            loss.backward()

            # optimize model parameters
            optimizer.step()

        # save epoch stats
        train_loss.append(np.mean(losses))
        train_acc.append(np.mean(acc))

        # set model to evaluation mode
        model.eval()

        # disable gradient calculation
        with torch.no_grad():
            losses, acc = [], []

            for X, y_true in tqdm(val_dl, desc=f'-> [val] epoch {epoch + 1} '):

                # send batch to right device
                X = X.to(device)
                y_true = y_true.to(device)

                # predict label scores
                y_pred = model(X)

                # compute loss
                loss = loss_func(y_pred, y_true)

                # compute accuracy
                gold = y_true.cpu().numpy()
                pred = np.argmax(y_pred.cpu().numpy(), axis=1)

                # accumulate for plotting
                losses.append(loss.cpu().item())
                acc.append(accuracy_score(gold, pred))
            # save epoch stats
            val_loss.append(np.mean(losses))
            val_acc.append(np.mean(acc))


    history = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }


    return model, history



def test_model(model, test_dl, y_true):
    # set model to evaluation mode
    model.eval()

    y_pred = []

    # disable gradient calculation
    with torch.no_grad():
        for X, _ in tqdm(test_dl):
            X = X.to(device)
            # predict one class per example
            y = torch.argmax(model(X), dim=1)
            # convert tensor to numpy array
            y_pred.append(y.cpu().numpy())

    # print results
    y_pred = np.concatenate(y_pred)
    print(classification_report(y_true, y_pred, target_names=agnews_labels))

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=agnews_labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(cmap='Blues', values_format='.2f', colorbar=False, ax=ax, xticks_rotation=45)


if __name__ == '__main__':
    train_df, val_df, test_df, vocabulary_size = get_split_data_with_features(agnews_train_file, agnews_test_file)

    # hyperparameters
    hyperparam = {
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 500,
        "shuffle": True,
        "n_epochs": 5,
        "input_dim": vocabulary_size,
        "hidden_dim": 50,
        "output_dim": len(agnews_labels),
        "dropout": 0.3,
    }

    train_dl, val_dl, test_dl = prepare_data(hyperparam, train_df, val_df, test_df)


    model, history = train_model(hyperparam, train_dl, val_dl)
    plot_train_history(hyperparam, history)

    test_model(model, test_dl, y_true=test_df['class index'] - 1)

