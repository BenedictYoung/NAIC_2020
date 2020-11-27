from utils import *
from functions import *
from torch.utils.data import Dataset


class data_set(Dataset):
    def __init__(self, H, pilot_num):
        super().__init__()
        self.H = H
        self.pilot_num = pilot_num

    def __getitem__(self, item):
        HH = self.H[item]
        SNRdb = np.random.uniform(8, 12)
        mode = np.random.randint(0, 3)
        bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        X = [bits0, bits1]
        YY = MIMO(X, HH, SNRdb, mode, self.pilot_num) / 20  
        XX = np.concatenate((bits0, bits1), 0)
        return Y2S(YY), XX, H2L(HH)

    def __len__(self):
        return len(self.H)


class channel_set(Dataset):
    def __init__(self, sample, label):
        super().__init__()
        self.sample = sample
        self.label = label

    def __getitem__(self, item):
        return self.sample[item, :, :, :], self.label[item, :]

    def __len__(self):
        return len(self.sample)
