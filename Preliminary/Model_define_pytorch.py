import os
import numpy as np
from torch.utils import data
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from collections import OrderedDict



A_num = np.random.normal(loc=0,scale=1/np.sqrt(512),size=(512,1024))
A_num = A_num.astype('float32')
# print(A_num)

# This part implement the quantization and dequantization operations.
# The output of the encoder must be the bitstream.
def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its four bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2)
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for four time.
        #b, c = grad_output.shape
        #grad_bit = grad_output.repeat(1, 1, ctx.constant) 
        #return torch.reshape(grad_bit, (-1, c * ctx.constant)), None
        grad_bit = grad_output.repeat_interleave(ctx.constant, dim=1)
        return grad_bit, None


class QuantizationLayer(nn.Module):

    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out


class DequantizationLayer(nn.Module):

    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out


class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out



class Encoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits = 128, quantization = True, nc = 2, num_Blocks = [2,2,2,2], trainable_nn = True):
        super(Encoder, self).__init__()
        self.feedback_bits=feedback_bits
        self.in_planes = 64
        self.z_dim = int(feedback_bits / self.B)
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, self.z_dim)
        self.sig = nn.Sigmoid()

        self.quantize = QuantizationLayer(self.B)
        self.quantization = quantization
        self.A = nn.Parameter(torch.tensor(A_num), requires_grad=False)
        self.trainable_nn = trainable_nn

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.trainable_nn:
            for i in self.parameters():
                i.requires_grad = True
        else:
            for i in self.parameters():
                i.requires_grad = False
        self.A.requires_grad = False
        x = x - 0.5
        x = x.reshape([-1, 1024])
        x = torch.transpose(x, 0, 1)
        x = torch.matmul(self.A, x)
        x = torch.transpose(x, 0, 1)
        x = x.reshape([-1, 2, 8, 32])
        x_term = x + 0.5
        x = x + 0.5

        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.sig(x)
        if self.quantization:
            out = self.quantize(x)
        else:
            out = x

        return out

    
def eta(r, rvar, theta):
    "implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)"
    # threshold = torch.tensor(0.).cuda()
    # theta = torch.max(theta, threshold)
    
    theta = F.relu(theta)
    lam = torch.sqrt(rvar)*theta
    # xhat = torch.sign(r) * torch.max(torch.abs(r) - lam,  torch.tensor(0.).cuda())
    xhat = torch.sign(r) * F.relu(torch.abs(r) - lam)
    dxdr = torch.mean(((torch.abs(r) - lam) > 0).float(), 0)
    return xhat, dxdr

class LAMP_Initial_Layer(nn.Module):

    def __init__(self, A):
        super(LAMP_Initial_Layer, self).__init__()
        self.A = A
        self.M = 512
        self.N = 1024
        self.OneOverM = torch.tensor(float(1) / self.M).cuda()
        self.NOverM = torch.tensor(float(self.N) / self.M).cuda()
        self.Bt = nn.Parameter(torch.transpose(self.A, 0, 1),requires_grad=False)
        self.theta = nn.Parameter(torch.tensor(1.),requires_grad=False)

    def forward(self, y):
        v_previous = y
        rvar = self.OneOverM * torch.sum(v_previous * v_previous, 0)
        xhat, deta = eta(torch.matmul(self.Bt, v_previous), rvar, self.theta)
        b = self.NOverM * deta
        v = y - torch.matmul(self.A, xhat) + b * v_previous
        return xhat, v


class LAMP_Single_Layer(nn.Module):

    def __init__(self, A, layer_index):
        super(LAMP_Single_Layer, self).__init__()
        self.M = 512
        self.N = 1024
        self.A = A
        self.layer_index = layer_index
        self.OneOverM = torch.tensor(float(1) / self.M).cuda()
        self.NOverM = torch.tensor(float(self.N) / self.M).cuda()
        self.Bt = nn.Parameter(torch.transpose(self.A, 0, 1),requires_grad=False)
        self.theta = nn.Parameter(torch.tensor(1.),requires_grad=False)

    def forward(self, out_previous):
        y = out_previous[0]
        xhat_previous = out_previous[1]
        v_previous = out_previous[2]
        r = xhat_previous + torch.matmul(self.Bt, v_previous)
        rvar = self.OneOverM * torch.sum(v_previous * v_previous, 0)
        xhat, deta = eta(r, rvar, self.theta)
        b = self.NOverM * deta
        v = y - torch.matmul(self.A, xhat) + b * v_previous
        out = [y, xhat, v]
        return out



class Decoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits = 128, quantization=True, nc = 2, num_Blocks = [3,4,23,3], Layer_num=8, trainable_nn = True, trainable_LAMP=False):
        super(Decoder, self).__init__()
        self.in_planes = 512
        self.feedback_bits = feedback_bits
        self.quantization = quantization
        self.dequantize = DequantizationLayer(self.B)
        z_dim = int(feedback_bits / self.B)

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=0.5)

        self.trainable_nn = trainable_nn
        self.trainable_LAMP = trainable_LAMP
        self.Layer_num=Layer_num
        self.A = nn.Parameter(torch.tensor(A_num), requires_grad=False)
        self.LAMP_Initial_Layer = LAMP_Initial_Layer(self.A)
        LAMP = OrderedDict()
        for layer_index in range(self.Layer_num):
            key = 'LAMP_Layer'+str(layer_index+1)
            LAMP[key] = LAMP_Single_Layer(self.A, layer_index+1)
        self.LAMP_Layer = nn.Sequential(LAMP)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.quantization:
            out = self.dequantize(x)
        else:
            out = x
        if self.trainable_nn:
            for i in self.parameters():
                i.requires_grad = True
        else:
            for i in self.parameters():
                i.requires_grad = False

        if self.trainable_LAMP:
            for i in (self.LAMP_Initial_Layer).parameters():
                i.requires_grad = True
            for i in self.LAMP_Layer.parameters():
                i.requires_grad = True

        else:
            for i in (self.LAMP_Initial_Layer).parameters():
                i.requires_grad = False
            for i in self.LAMP_Layer.parameters():
                i.requires_grad = False

        self.A.requires_grad = False

        x = self.linear(out)
        x = x.view(out.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 2, 8, 32)
        out = x
        out_term = x

        out = out - 0.5
        out = out.reshape([-1, 512])
        out = torch.transpose(out, 0, 1)
        xhat, v = self.LAMP_Initial_Layer(out)
        out_list = self.LAMP_Layer([out, xhat, v])
        xhat = out_list[1]
        out = torch.transpose(xhat,0,1)
        out = out.reshape([-1, 2, 16, 32])
        out = out + 0.5

        return out


# Note: Do not modify following class and keep it in your submission.
# feedback_bits is 128 by default.
class AutoEncoder(nn.Module):

    def __init__(self,feedback_bits):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(feedback_bits)
        self.decoder = Decoder(feedback_bits)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)

        return out


def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse

def NMSE_cuda(x, x_hat):
    x_real = x[:, 0, :, :].view(len(x),-1) - 0.5
    x_imag = x[:, 1, :, :].view(len(x),-1) - 0.5
    x_hat_real = x_hat[:, 0, :, :].view(len(x_hat), -1) - 0.5
    x_hat_imag = x_hat[:, 1, :, :].view(len(x_hat), -1) - 0.5
    power = torch.sum(x_real**2 + x_imag**2, axis=1)
    mse = torch.sum((x_real-x_hat_real)**2 + (x_imag-x_hat_imag)**2, axis=1)
    nmse = mse/power
    return nmse
    
class NMSELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x, x_hat)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse) 
        else:
            nmse = torch.sum(nmse)
        return nmse

def Score(NMSE):
    score = 1 - NMSE
    return score


# dataLoader
class DatasetFolder(Dataset):

    def __init__(self, matData):
        self.matdata = matData

    def __len__(self):
        return self.matdata.shape[0]
    
    def __getitem__(self, index):
        return self.matdata[index] #, self.matdata[index]
