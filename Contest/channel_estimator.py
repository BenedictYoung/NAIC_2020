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
parser.add_argument('--bs', type=int, default=1000)
parser.add_argument('--cuda_list', type=str, default='3, 4, 5')
args = parser.parse_args()

models = [34, 50, 121, 169, 201]
pilots = [8, 32]

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_list

for pn in pilots:
    if pn == 32:
        Y = np.loadtxt('./DataSet/Y_1.csv', dtype=np.float, delimiter=',')
    else:
        Y = np.loadtxt('./DataSet/Y_2.csv', dtype=np.float, delimiter=',')
        
    for m in models:
        if m == 34:
            estimator = resnet34()
        elif m == 50:
            estimator = resnet50()
        elif m == 121:
            estimator = densenet121()
        elif m == 169:
            estimator = densenet169()
        else:
            estimator = densenet201()

        estimator = torch.nn.DataParallel(estimator).cuda()
        estimator_path = './Networks/Y2H_p' + str(pn) + '_' + str(m) + '.pth'
        estimator.load_state_dict(torch.load(estimator_path))
        
        print("channel estimating for model:{} pilot_number:{}".format(m, pn))
        with torch.no_grad():
            estimator.eval()
            sample = np.reshape(Y, (-1, 1, 8, 256), order='F')
            sample = torch.as_tensor(sample, dtype=torch.float, device='cuda')
        
            predict = []
            for i in range(10000 // args.bs):
                print("current sample[{}]-[{}]".format(i * args.bs, (i + 1) * args.bs))
                sample_batch = sample[i * args.bs: (i + 1) * args.bs, :, :, :]
                predict_batch = estimator(sample_batch)
                predict_batch = predict_batch.detach().cpu().numpy()
                predict.append(predict_batch)
        
            hypothesis = np.concatenate(predict)  # Float NS * 2 * 4 * 32
            Hf = Ht2Hf(hypothesis)
            Hf = HfComplex2Float(Hf)
            Y = np.reshape(Y, (-1, 2, 2, 2, 256), order='F')
            Yd = Y[:, :, 1, :, :]
            sample = np.concatenate((Yd, Hf), axis=2)  # Ns * 2 * 6 * 256 Float
            sample = np.reshape(sample, (-1, 1, 12, 256))  # Ns * 1 * 12 * 256 Float
            H_path = './Temporary/H_hat' + str(pn) + '_' + str(m)
            np.save(H_path, sample)
            print("H_hat for model:{} pilot_number:{} has been saved".format(m, pn))
            print()
