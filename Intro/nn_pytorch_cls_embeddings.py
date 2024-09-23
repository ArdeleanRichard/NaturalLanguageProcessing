import numpy as np
import torch

from sklearn.metrics import accuracy_score, classification_report
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.constants import agnews_labels, device
from utils.data_agnews_embeddings import MyDataset, get_split_data_with_embeddings
from utils.model_aux import plot_train_history


def get_dataloaders_from_dataframes(hyperparam, train_df, val_df, test_df):
    train_ds = MyDataset(train_df['token ids'], train_df['class index'] - 1)
    train_dl = DataLoader(train_ds, batch_size=hyperparam['batch_size'], shuffle=hyperparam['shuffle'])

    val_ds = MyDataset(val_df['token ids'], val_df['class index'] - 1)
    val_dl = DataLoader(val_ds, batch_size=hyperparam['batch_size'], shuffle=hyperparam['shuffle'])

    test_ds = MyDataset(test_df['token ids'], test_df['class index'] - 1)
    test_dl = DataLoader(test_ds, batch_size=hyperparam['batch_size'])

    return train_dl, val_dl, test_dl

class Model(nn.Module):
    def __init__(self, vectors, pad_id, hidden_dim, output_dim, dropout):
        super().__init__()
        # embeddings must be a tensor
        if not torch.is_tensor(vectors):
            vectors = torch.tensor(vectors)

        # keep padding id
        self.padding_idx = pad_id

        # embedding layer
        self.embs = nn.Embedding.from_pretrained(vectors, padding_idx=pad_id)

        # feedforward layers
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(vectors.shape[1], hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # get boolean array with padding elements set to false
        not_padding = torch.isin(x, self.padding_idx, invert=True)

        # get lengths of examples (excluding padding)
        lengths = torch.count_nonzero(not_padding, axis=1)

        # get embeddings
        x = self.embs(x)

        # calculate means
        x = x.sum(dim=1) / lengths.unsqueeze(dim=1)

        # pass to rest of the model
        output = self.layers(x)

        # calculate softmax if we're not in training mode
        # if not self.training:
        #    output = F.softmax(output, dim=1)

        return output

def train_model(hyperparam, train_dl, val_dl):
    model = Model(
        hyperparam['vectors'],
        hyperparam['pad_id'],
        hyperparam['hidden_dim'],
        hyperparam['output_dim'],
        hyperparam['dropout']
    ).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparam['lr'], weight_decay=hyperparam['weight_decay'])

    train_loss = []
    train_acc = []

    val_loss = []
    val_acc = []

    # train the model
    for epoch in range(hyperparam['n_epochs']):
        losses = []
        gold = []
        pred = []
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
            # accumulate for plotting
            losses.append(loss.detach().cpu().item())
            gold.append(y_true.detach().cpu().numpy())
            pred.append(np.argmax(y_pred.detach().cpu().numpy(), axis=1))
            # backpropagate
            loss.backward()
            # optimize model parameters
            optimizer.step()

        train_loss.append(np.mean(losses))
        train_acc.append(accuracy_score(np.concatenate(gold), np.concatenate(pred)))

        model.eval()
        with torch.no_grad():
            losses = []
            gold = []
            pred = []
            for X, y_true in tqdm(val_dl, desc=f'-> [val] epoch {epoch + 1} '):
                X = X.to(device)
                y_true = y_true.to(device)
                y_pred = model(X)
                loss = loss_func(y_pred, y_true)
                losses.append(loss.cpu().item())
                gold.append(y_true.cpu().numpy())
                pred.append(np.argmax(y_pred.cpu().numpy(), axis=1))
            val_loss.append(np.mean(losses))
            val_acc.append(accuracy_score(np.concatenate(gold), np.concatenate(pred)))

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




if __name__ == '__main__':
    glove, train_df, val_df, test_df = get_split_data_with_embeddings()

    # hyperparameters
    hyperparam = {
        "lr": 1e-3,
        "weight_decay": 0,
        "batch_size": 500,
        "shuffle": True,
        "n_epochs": 5,
        "hidden_dim": 50,
        "output_dim": len(agnews_labels),
        "dropout": 0.1,
        "vectors": glove.vectors,
        "pad_id": glove.key_to_index['[PAD]']
    }

    train_dl, val_dl, test_dl = get_dataloaders_from_dataframes(hyperparam, train_df, val_df, test_df)

    model, history = train_model(hyperparam, train_dl, val_dl)
    plot_train_history(hyperparam, history)

    test_model(model, test_dl, y_true=test_df['class index'] - 1)