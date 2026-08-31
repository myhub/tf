import torch
import os
import random
from torch.utils.data import Dataset
import sys
from io import StringIO

TEST_SIZE = 100000

class NlpA60(Dataset): 
    def __init__(self, is_training=True):
        
        self.alphabet = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+', "="]

        self.seq_len = 192
        self.aout_len = 64

        self.is_training = is_training
        
        self.char2idx = {}
        for i, c in enumerate(self.alphabet):
            self.char2idx[c] = i

    def __len__(self):
        if self.is_training:
            return 2048 * 8
        else:
            return TEST_SIZE
    
    def __getitem__(self, index):
        # 因为数据集规模足够大，测试集出现在训练集的概率极低，所以此处不区分训练集/测试集
        _ = index
        
        m1 = random.randrange(10**60)
        m2 = random.randrange(10**60)
        is_add = random.random() < 0.5

        if is_add:
            ain = f"{m1}+{m2}="
            aout = f"{m1+m2}"
        else:
            ain = f"{m1}-{m2}="
            aout = f"{m1-m2}"
        
        ain += ' '*self.aout_len

        ain = [0]*(self.seq_len - len(ain)) + [self.char2idx[c] for c in ain] 
        aout = [0]*(self.aout_len - len(aout)) + [self.char2idx[c] for c in aout]
        
        assert len(ain) == self.seq_len 
        assert len(aout) == self.aout_len 
        
        return torch.LongTensor(ain), torch.LongTensor(aout)


if __name__ == '__main__':
    ds = NlpA60(is_training=False)

    for i in range(len(ds)):
        ain, aout = ds[i]
        print(' | ', end="")
        for idx in ain:
            c = ds.alphabet[idx]
            if c == ' ': continue
            print(c, end='')
        print(' | ', end="")
        for idx in aout:
            c = ds.alphabet[idx]
            if c == ' ': continue
            print(c, end='')
        print(' | ')