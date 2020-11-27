import os
import sys
import time
import torch
import struct
import argparse
import numpy as np

from res_net import *
from dense_net import *
from functions import *
from data_generator import *

from torch.utils.data import DataLoader

# args parameters
parser = argparse.ArgumentParser()
parser.add_argument('--pn', type=int, default=8)
parser.add_argument('--m', type=int, default=34)
parser.add_argument('--bs', type=int, default=512)
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

if args.m == 34:
    model = resnet34()
elif args.m == 50:
    model = resnet50()
elif args.m == 121:
    model = densenet121()
elif args.m == 169:
    model = densenet169()
else:
    model = densenet201()

model = torch.nn.DataParallel(model).cuda()
path = './Networks/Y2H_p' + str(args.pn) + '_' + str(args.m) + '.pth'

if args.load:
    model.load_state_dict(torch.load(path))

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), args.lr)
epochs = 100

loss_min = 1
print("training for estimator...")
for epoch in range(epochs):

    model.train()
    for step, (Y, X, H) in enumerate(train_data_loader):
        sample = torch.as_tensor(Y, dtype=torch.float, device='cuda')
        label = torch.as_tensor(H, dtype=torch.float, device='cuda')
        predict = model(sample)

        loss = loss_func(predict, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('epoch: [{}/{}][{}/{}]\t train_loss {:.4f}'.format(epoch, epochs, step, len(train_data_loader),
                                                                     loss.item()),
                  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    model.eval()
    with torch.no_grad():
        predict_collect = []
        label_collect = []
        for step, (Y, X, H) in enumerate(valid_data_loader):
            sample = torch.as_tensor(Y, dtype=torch.float, device='cuda')
            label = torch.as_tensor(H, dtype=torch.float, device='cuda')
            predict = model(sample)

            predict_collect.append(predict.detach().cpu())
            label_collect.append(label.detach().cpu())

        loss = loss_func(torch.cat(predict_collect), torch.cat(label_collect))
        valid_loss = loss.item()
        print('valid_loss {:.4f}'.format(valid_loss), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if valid_loss < loss_min:
            loss_min = valid_loss
            torch.save(model.state_dict(), path)

    optimizer.param_groups[0]['lr'] *= 0.98
