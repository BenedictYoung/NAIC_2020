import numpy as np
import struct
import torch

models = [34, 50, 121, 169, 201]


def read_data(path):
    X = open(path, 'rb')
    Y = struct.unpack('?' * 10000 * 1024, X.read(2 * 10000 * 1024))
    Y = np.reshape(Y, [10000, 1024])
    return Y


def get_final(index):
    data = []
    for m in models:
        data_batch = read_data('./Temporary/X_pre_' + str(index) + '_' + str(m) + '.bin')
        data_batch = torch.tensor(data_batch, dtype=torch.long)
        data.append(data_batch)
    data = torch.stack(data, dim=0)
    X_final, X_indices = torch.mode(data, dim=0)
    X_final = X_final.numpy().astype(np.bool)
    return X_final


X1 = get_final(1)
X1.tofile('./Result/X_pre_1.bin')

X2 = get_final(2)
X2.tofile('./Result/X_pre_2.bin')

print("X_hat after voting has been saved")
