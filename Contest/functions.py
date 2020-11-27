import numpy as np
import time
import torch


def Y2S(Y):
    Y = np.reshape(Y, (1, 8, 256), order='F')
    return Y


def H2L(H):
    Ht = np.zeros(shape=(2, 4, 32), dtype=np.float)
    Ht[0, :, :] = H.real
    Ht[1, :, :] = H.imag
    return Ht


def Ht2Hf(H):
    Ht = H[:, 0, :, :] + 1j * H[:, 1, :, :]  # Complex NS * 4 * 32
    Hf = np.fft.fft(Ht, 256) / 20
    Hf = np.reshape(Hf, (-1, 4, 256), order='F')  # Complex Ns * 4 * 256
    return Hf


def Y2Yd(Y):
    Y = np.reshape(Y, (-1, 2, 2, 2, 256), order='F')
    Y = Y[:, 0, :, :, :] + 1j * Y[:, 1, :, :, :]
    Yd = Y[:, 1, :, :]
    return Yd


def HfComplex2Float(H):
    Hf = np.zeros(shape=[H.shape[0], 2, 4, 256], dtype=np.float)  # Float Ns * 2 * 4 * 256
    Hf[:, 0, :, :] = H.real
    Hf[:, 1, :, :] = H.imag
    return Hf


def CalAccuracy(X_hat, X):
    X_hat = X_hat.detach().cpu().numpy()
    X = X.detach().cpu().numpy()
    X_hat = np.array(np.floor(X_hat + 0.5), dtype=np.bool)
    same = (X == X_hat)
    return (np.sum(same, axis=1) / same.shape[1]).mean()
