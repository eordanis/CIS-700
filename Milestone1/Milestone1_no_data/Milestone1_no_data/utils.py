import os
import matplotlib.pyplot as plt
import numpy as np


def create_directories(path):
    if not os.path.exists('{}'.format(path)):
        os.makedirs('{}/models'.format(path))
        os.makedirs('{}/images'.format(path))
        os.makedirs('{}/plots'.format(path))

    else:
        print("path already exists")


def get_batch(data, batch_size):
    idx = np.random.randint(0, data.shape[0], batch_size)
    real_seqs = data[idx]
    return real_seqs


def sample_image(gan, epoch, X_train, path):
    fig, axarray = plt.subplots(4, 1, figsize=(10, 12))
    idx = np.random.randint(0, X_train.shape[0])

    axarray[0].plot(X_train[idx, 0, :], c='r')
    axarray[0].plot(X_train[idx, 1, :], c='b')
    axarray[0].plot(X_train[idx, 2, :], c='g')

    axarray[0].set_title("Training data")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    for i, ax in enumerate(axarray.flatten()):
        if i != 0:
            xx = gan.generate()
            ax.plot(xx[0, 0, :], c='r', label="acc.x")
            ax.plot(xx[0, 1, :], c='b', label="acc.y")
            ax.plot(xx[0, 2, :], c='g', label="acc.z")

            ax.set_title("Generated example {}".format(i))
    plt.savefig('{}/images/sample{}.png'.format(path, epoch), transparent=True)
    plt.close()


def plot_4dim(data):
    idx = 0
    plt.plot(data[idx, 0, :, :])
    plt.plot(data[idx, 1, :, :])
    plt.plot(data[idx, 2, :, :])
