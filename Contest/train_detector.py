import os
import sys
import time
import torch
import struct
import argparse
import numpy as np

from res_net import *
from functions import *
from data_generator import *

from torch.utils.data import DataLoader

# args parameters
parser = argparse.ArgumentParser()
parser.add_argument('--pn', type=int, default=8)
parser.add_argument('--m', type=int, default=34)
parser.add_argument('--bs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--load', default=True, action='store_true')
parser.add_argument('--cuda_list', type=str, default='2, 3, 4, 5, 6, 7')
args = parser.parse_args()

# load data
data1 = open('./DataSet/H.bin', 'rb')
H1 = struct.unpack('f' * 2 * 2 * 2 * 32 * 320000, data1.read(4 * 2 * 2 * 2 * 32 * 320000))
H1 = np.reshape(H1, [320000, 2, 4, 32])
H_train = H1[:, 1, :, :] + 1j * H1[:, 0, :, :]

data2 = open('./DataSet/H_val.bin', 'rb')
H2 = struct.unpack('f' * 2 * 2 * 2 * 32 * 2000, data2.read(4 * 2 * 2 * 2 * 32 * 2000))
H2 = np.reshape(H2, [2000, 2, 4, 32])
H_valid = H2[:, 1, :, :] + 1j * H2[:, 0, :, :]

train_data_set = data_set(H=H_train, pilot_num=args.pn)
train_data_loader = DataLoader(dataset=train_data_set, batch_size=args.bs, shuffle=True, num_workers=8)

valid_data_set = data_set(H=H_valid, pilot_num=args.pn)
valid_data_loader = DataLoader(dataset=valid_data_set, batch_size=args.bs, shuffle=True, num_workers=8)

print("data loaded")

# define hyper parameters & model
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_list

model = resnet18(1024)
model = torch.nn.DataParallel(model).cuda()
path = './Networks/H2X_p' + str(args.pn) + '_' + str(args.m) + '.pth'

if args.load:
    model.load_state_dict(torch.load(path))

loss_func = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), args.lr)
epochs = 100

acc_max = 0.0
print("training for detector...")
for epoch in range(epochs):

    model.train()
    for step, (Y, X, H) in enumerate(train_data_loader):

        H = H.detach().cpu().numpy()  # Float Ns * 2 * 4 * 32
        Hf = Ht2Hf(H)  # Complex Ns * 4 * 256
        Hf = HfComplex2Float(Hf)  # Float Ns * 2 * 4 * 256

        Y = Y.detach().cpu().numpy()
        Y = np.reshape(Y, (-1, 2, 2, 2, 256), order='F')
        Yd = Y[:, :, 1, :, :]  # Float Ns * 2 * 2 * 256

        sample = np.concatenate((Yd, Hf), axis=2)  # Ns * 2 * 6 * 256 Float
        sample = np.reshape(sample, (-1, 1, 12, 256))  # Ns * 1 * 12 * 256 Float

        sample = torch.as_tensor(sample, dtype=torch.float, device='cuda')
        label = torch.as_tensor(X, dtype=torch.float, device='cuda')
        predict = model(sample)

        loss = loss_func(predict, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = CalAccuracy(predict, label)
        if step % 100 == 0:
            print('epoch: [{}/{}][{}/{}]\t\t train_acc {:.4f}'.format(epoch, epochs, step, len(train_data_loader),
                                                                      train_acc),
                  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    model.eval()
    with torch.no_grad():
        predict_collect = []
        label_collect = []
        for step, (Y, X, H) in enumerate(valid_data_loader):
            H = H.detach().cpu().numpy()  # Float NS * 2 * 4 * 32
            Hf = Ht2Hf(H)  # Complex Ns * 4 * 256
            Hf = HfComplex2Float(Hf)  # Float Ns * 2 * 4 * 256

            Y = Y.detach().cpu().numpy()
            Y = np.reshape(Y, (-1, 2, 2, 2, 256), order='F')
            Yd = Y[:, :, 1, :, :]  # Float Ns * 2 * 2 * 256

            sample = np.concatenate((Yd, Hf), axis=2)  # Ns * 2 * 6 * 256 Float
            sample = np.reshape(sample, (-1, 1, 12, 256))  # Ns * 1 * 12 * 256 Float

            sample = torch.as_tensor(sample, dtype=torch.float, device='cuda')
            label = torch.as_tensor(X, dtype=torch.float, device='cuda')
            predict = model(sample)

            predict_collect.append(predict)
            label_collect.append(label)

        valid_acc = CalAccuracy(torch.cat(predict_collect), torch.cat(label_collect))
        print('valid_acc {:.4f}'.format(valid_acc), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if valid_acc > acc_max:
            acc_max = valid_acc
            torch.save(model.state_dict(), path)

    optimizer.param_groups[0]['lr'] *= 0.95
