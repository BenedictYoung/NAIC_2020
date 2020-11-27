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
parser.add_argument('--bs', type=int, default=1000)
parser.add_argument('--cuda_list', type=str, default='2, 3, 4, 5, 6, 7')
args = parser.parse_args()

models = [34, 50, 121, 169, 201]
pilots = [8, 32]

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_list

detector = resnet18(1024)
detector = torch.nn.DataParallel(detector).cuda()

for pn in pilots:
    for m in models:

        input_path = "./Temporary/H_hat" + str(pn) + "_" + str(m) + ".npy"
        sample = np.load(input_path)
        detector_path = './Networks/H2X_p' + str(pn) + '_' + str(m) + '.pth'
        detector.load_state_dict(torch.load(detector_path))

        print("signal detecting for model:{} pilot_number:{}".format(m, pn))

        with torch.no_grad():
            detector.eval()
            sample = torch.as_tensor(sample, dtype=torch.float, device='cuda')

            predict = []
            for i in range(10000 // args.bs):
                print("current sample[{}]-[{}]".format(i * args.bs, (i + 1) * args.bs))
                sample_batch = sample[i * args.bs: (i + 1) * args.bs, :, :, :]
                predict_batch = detector(sample_batch)
                predict_batch = predict_batch.detach().cpu().numpy()
                predict.append(predict_batch)

            X_hat = np.concatenate(predict)

            if pn == 32:
                X = np.array(np.floor(X_hat + 0.5), dtype=np.bool)
                X_path = './Temporary/X_pre_1_' + str(m) + '.bin'
                X.tofile(X_path)

            if pn == 8:
                X = np.array(np.floor(X_hat + 0.5), dtype=np.bool)
                X_path = './Temporary/X_pre_2_' + str(m) + '.bin'
                X.tofile(X_path)

            print("X_hat for model:{} pilot_number:{} has been saved".format(m, pn))
            print()
