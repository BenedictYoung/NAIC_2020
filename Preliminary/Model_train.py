'''
seefun . Aug 2020.
github.com/seefun | kaggle.com/seefun
'''

import numpy as np
import h5py
import torch
import os
import torch.nn as nn
import random
import time
from Model_define_pytorch import AutoEncoder, DatasetFolder, NMSELoss

# Parameters for training
gpu_list = '0,1,2,3'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

batch_size = 256
epochs = 200
learning_rate = 1e-4 # bigger to train faster
num_workers = 4
print_freq = 100
train_test_ratio = 0.8

# parameters for data
feedback_bits = 128
img_height = 16
img_width = 32
img_channels = 2

load_flag = 1
train_flag = 2

# Model construction
model = AutoEncoder(feedback_bits).cuda()

if load_flag==1:
  modelSave1 = './Modelsave/encoder.pth.tar'
  model.encoder.load_state_dict(torch.load(modelSave1)['state_dict'])
  print("weight loaded")
  
  modelSave2 = './Modelsave/decoder.pth.tar'
  model.decoder.load_state_dict(torch.load(modelSave2)['state_dict'])
  print("weight loaded")

if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda() # model.module
else:
    model = model.cuda()

criterion = NMSELoss(reduction='mean') # nn.MSELoss()
criterion_test = NMSELoss(reduction='sum')

try:
    model.encoder.quantization = True
    model.decoder.quantization = True
except:
    model.module.encoder.quantization = True
    model.module.decoder.quantization = True


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Data loading
data_load_address = './data'
mat = h5py.File(data_load_address + '/H_train.mat', 'r')
data = np.transpose(mat['H_train'])  # shape=(320000, 1024)
data = data.astype('float32')
data = np.reshape(data, [len(data), img_channels, img_height, img_width])

# split data for training(80%) and validation(20%)
np.random.shuffle(data)
start = int(data.shape[0] * train_test_ratio)
x_train, x_test = data[:start], data[start:]


# dataLoader for training
train_dataset = DatasetFolder(x_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

# dataLoader for training
test_dataset = DatasetFolder(x_test)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


best_loss = 1
for epoch in range(epochs):           
    # model training
    model.train()
    if load_flag==0:
      if epoch < epochs//10:
          try:
              model.encoder.quantization = False
              model.decoder.quantization = False
          except:
              model.module.encoder.quantization = False
              model.module.decoder.quantization = False
      else:
          try:
              model.encoder.quantization = True
              model.decoder.quantization = True
          except:
              model.module.encoder.quantization = True
              model.module.decoder.quantization = True

    if epoch==1:
      for name, param in model.named_parameters():
        if param.requires_grad:
           print("requires_grad: True ", name)
        else:
           print("requires_grad: False ", name)
  
    if epoch == epochs//4 * 3:
        optimizer.param_groups[0]['lr'] =  optimizer.param_groups[0]['lr'] * 0.25
       

    
    for i, input in enumerate(train_loader):

        if train_flag==0:
          model.encoder.trainable_nn = True
          model.decoder.trainable_LAMP = False
          model.decoder.trainable_nn = True

          input = input.cuda()
          output, loss_term = model(input)      
          loss_final = criterion(output,input)
          loss = loss_term
          # loss = loss_term
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
          
        if train_flag==1:
        
          model.encoder.trainable_nn = False
          model.decoder.trainable_LAMP = True
          model.decoder.trainable_nn = True


          input = input.cuda()
          output,loss_term = model(input)
          loss_final = criterion(output,input)
          loss = loss_term
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
                   
        if train_flag==2:
        
          model.encoder.trainable_nn = False
          model.decoder.trainable_LAMP = True
          model.decoder.trainable_nn = True


          input = input.cuda()
          output,loss_term = model(input)
          loss_final = criterion(output,input)
          loss = loss_final
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()



        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss_final {loss_final:.4f}\t''Loss_term {loss_term:.4f}'.format(
                epoch, i, len(train_loader), loss_final=loss_final.item(), loss_term=loss_term.item()))

    model.eval()
    try:
        model.encoder.quantization = True
        model.decoder.quantization = True
    except:
        model.module.encoder.quantization = True
        model.module.decoder.quantization = True
    total_loss = 0
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            # convert numpy to Tensor
            input = input.cuda()
            output, loss_term = model(input)
            total_loss += criterion_test(output, input).item()
            # total_loss += loss_term
        average_loss = total_loss / (len(test_dataset))
        print('NMSE %.4f'%average_loss)
        if average_loss < best_loss:
            # model save
            # save encoder
            modelSave1 = './Modelsave/encoder.pth.tar'
            try:
                torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1, _use_new_zipfile_serialization=False)
            except:
                torch.save({'state_dict': model.module.encoder.state_dict(), }, modelSave1, _use_new_zipfile_serialization=False)
            # save decoder
            modelSave2 = './Modelsave/decoder.pth.tar'
            try:
                torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2, _use_new_zipfile_serialization=False)
            except:
                torch.save({'state_dict': model.module.decoder.state_dict(), }, modelSave2, _use_new_zipfile_serialization=False)
            print('Model saved!')
            best_loss = average_loss
