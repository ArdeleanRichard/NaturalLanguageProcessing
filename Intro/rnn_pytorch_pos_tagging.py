import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch._dynamo.trace_rules import np
from torch.nn.utils.rnn import pad_sequence
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.constants import device
from utils.data_ancora import MyDataset, get_split_dataframes
from utils.model_aux import plot_train_history


def collate_fn(batch, hyperparam):
    # separate xs and ys
    xs, ys = zip(*batch)
    # get lengths
    lengths = [len(x) for x in xs]

    # pad sequences
    x_padded = pad_sequence(xs, batch_first=True, padding_value=hyperparam["pad_tok_id"])
    y_padded = pad_sequence(ys, batch_first=True, padding_value=hyperparam["pad_tag_id"])

    # return padded
    return x_padded, y_padded, lengths


def prepare_data(hyperparam, train_df, val_df, test_df):
    train_ds = MyDataset(train_df['word ids'], train_df['tag ids'])
    train_dl = DataLoader(train_ds,
                          batch_size=hyperparam['batch_size'], shuffle=hyperparam['shuffle'],
                          collate_fn=lambda x: collate_fn(x, hyperparam))

    val_ds = MyDataset(val_df['word ids'], val_df['tag ids'])
    val_dl = DataLoader(val_ds,
                        batch_size=hyperparam['batch_size'], shuffle=hyperparam['shuffle'],
                        collate_fn=lambda x: collate_fn(x, hyperparam))

    test_ds = MyDataset(test_df['word ids'], test_df['tag ids'])
    test_dl = DataLoader(test_ds,
                         batch_size=hyperparam['batch_size'], shuffle=hyperparam['shuffle'],
                         collate_fn=lambda x: collate_fn(x, hyperparam))

    return train_dl, val_dl, test_dl

class Model(nn.Module):
    def __init__(self, vectors, hidden_size, num_layers, bidirectional, dropout, output_size):
        super().__init__()

        # ensure vectors is a tensor
        if not torch.is_tensor(vectors):
            vectors = torch.tensor(vectors)

        # init embedding layer
        self.embedding = nn.Embedding.from_pretrained(embeddings=vectors)

        # init lstm
        self.lstm = nn.LSTM(
            input_size=vectors.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True)

        # init dropout
        self.dropout = nn.Dropout(dropout)

        # init classifier
        self.classifier = nn.Linear(
            in_features=hidden_size * 2 if bidirectional else hidden_size,
            out_features=output_size)

    def forward(self, x_padded, x_lengths):
        # get embeddings
        output = self.embedding(x_padded)
        output = self.dropout(output)

        # pack data before lstm
        packed = pack_padded_sequence(output, x_lengths, batch_first=True, enforce_sorted=False)
        packed, _ = self.lstm(packed)

        # unpack data before rest of model
        output, _ = pad_packed_sequence(packed, batch_first=True)
        output = self.dropout(output)
        output = self.classifier(output)

        return output


def train_model(hyperparam, train_dl, val_dl):
    model = Model(
        hyperparam['vectors'],
        hyperparam['hidden_size'],
        hyperparam['num_layers'],
        hyperparam['bidirectional'],
        hyperparam['dropout'],
        hyperparam['output_size'],
    ).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparam['lr'], weight_decay=hyperparam['weight_decay'])

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    
    # train the model
    # One notable difference is that the output of this model has
    # three dimensions instead of two: number of examples, number of tokens,
    # and number of POS tag scores. Thus, we have to reshape the
    # output to pass it to the loss function. Additionally, we need to discard
    # the padding before computing the loss. We reshape the gold tag ids
    # using the torch.flatten() function, to transform the 2-dimensional
    # tensor of shape (n_examples, n_tokens) to a 1-dimensional tensor with
    # n_examples * n_tokens elements. The predictions are reshaped using
    # the view(-1, output_size) method. By passing two arguments we are
    # stipulating that we want two dimensions. The second dimension will be
    # of size output_size. The -1 indicates that the first dimension should be
    # inferred from the size of the tensor. This means that for a tensor of shape
    # (n_examples, n_tokens, output_size) we will get a tensor of shape
    # (n_examples * n_tokens, output_size). Then, we use a Boolean mask
    # to discard the elements corresponding to the padding. This way, the loss
    # function will consider each actual word individually, as if the whole batch
    # was just one big sentence. Note that treating a mini-batch as a single
    # virtual sentence does affect the evaluation results.
    for epoch in range(hyperparam['n_epochs']):
        losses, acc = [], []
        model.train()
        for x_padded, y_padded, lengths in tqdm(train_dl, desc=f'[train] epoch {epoch + 1} '):
            # clear gradients
            model.zero_grad()

            # send batch to right device
            x_padded = x_padded.to(device)
            y_padded = y_padded.to(device)

            # predict label scores
            y_pred = model(x_padded, lengths)

            # reshape output
            y_true = torch.flatten(y_padded)
            y_pred = y_pred.view(-1, hyperparam["output_size"])
            mask = y_true != pad_tag_id
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            # compute loss
            loss = loss_func(y_pred, y_true)

            # accumulate for plotting
            gold = y_true.detach().cpu().numpy()
            pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
            losses.append(loss.detach().cpu().item())
            acc.append(accuracy_score(gold, pred))

            # backpropagate
            loss.backward()

            # optimize model parameters
            optimizer.step()

        train_loss.append(np.mean(losses))
        train_acc.append(np.mean(acc))

        model.eval()
        with torch.no_grad():
            losses, acc = [], []
            for x_padded, y_padded, lengths in tqdm(val_dl, desc=f'-> [val] epoch {epoch + 1} '):
                x_padded = x_padded.to(device)
                y_padded = y_padded.to(device)
                y_pred = model(x_padded, lengths)
                y_true = torch.flatten(y_padded)
                y_pred = y_pred.view(-1, hyperparam["output_size"])
                mask = y_true != hyperparam["pad_tag_id"]
                y_true = y_true[mask]
                y_pred = y_pred[mask]
                loss = loss_func(y_pred, y_true)
                gold = y_true.cpu().numpy()
                pred = np.argmax(y_pred.cpu().numpy(), axis=1)
                losses.append(loss.cpu().item())
                acc.append(accuracy_score(gold, pred))
            val_loss.append(np.mean(losses))
            val_acc.append(np.mean(acc))
        
    history = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }

    return model, history



def test_model(hyperparam, model, test_dl):

    model.eval()
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for x_padded, y_padded, lengths in tqdm(test_dl):
            x_padded = x_padded.to(device)
            y_pred = model(x_padded, lengths)
            y_true = torch.flatten(y_padded)
            y_pred = y_pred.view(-1, hyperparam["output_size"])
            mask = y_true != hyperparam["pad_tag_id"]
            y_true = y_true[mask]
            y_pred = torch.argmax(y_pred[mask], dim=1)
            all_y_true.append(y_true.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    target_names = hyperparam["index_to_tag"][:-2]
    print(classification_report(y_true, y_pred, target_names=target_names))

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names,
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(
        cmap='Blues',
        values_format='.2f',
        colorbar=False,
        ax=ax,
        xticks_rotation=45,
    )
    plt.show()


if __name__ == '__main__':
    glove, train_df, val_df, test_df, index_to_tag, pad_tok_id, pad_tag_id = get_split_dataframes()

    # hyperparameters
    hyperparam = {
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 100,
        "shuffle": True,
        "n_epochs": 10,
        "vectors": glove.vectors,
        "hidden_size": 100,
        "num_layers": 2,
        "bidirectional": True,
        "dropout": 0.1,
        "output_size": len(index_to_tag),
        "index_to_tag": index_to_tag,
        "pad_tag_id": pad_tag_id,
        "pad_tok_id": pad_tok_id,
    }

    train_dl, val_dl, test_dl = prepare_data(hyperparam, train_df, val_df, test_df)

    model, history = train_model(hyperparam, train_dl, val_dl)
    plot_train_history(hyperparam, history)

    test_model(hyperparam, model, test_dl)

    # The results indicate that our POS tagger obtains an overall accuracy
    # of 97%, which is in line with state-of-the-art approaches!
    # This is encouraging considering that our approach does not include the CRF layer we
    # discussed in Chapter 10. We challenge the reader to add this layer,
    # https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
    # and experiment with this architecture for other sequence tasks such as
    # named entity recognition.