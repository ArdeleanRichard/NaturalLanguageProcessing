import numpy as np
from matplotlib import pyplot as plt


def plot_train_history(hyperparam, history):
    time = np.arange(hyperparam["n_epochs"]) + 1

    plt.figure()
    plt.plot(time, history['train_loss'])
    plt.plot(time, history['val_loss'])
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(time, history['train_acc'])
    plt.plot(time, history['val_acc'])
    plt.legend(['train_acc', 'val_acc'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.show()
    plt.close()