import torch
import os
import random
from torch.utils.data import Dataset
import sys
from io import StringIO

TEST_SIZE = 10000

class NlpM3(Dataset): 
    def __init__(self, is_training=True):
        self.a_z = []
   
        self.alphabet = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '*', "="]

        self.seq_len = 20
        self.aout_len = 8
        self.is_training = is_training

        self.char2idx = {}
        for i, c in enumerate(self.alphabet):
            self.char2idx[c] = i

        random.seed(2026)

        self.idx_list = list(range(1000000))
        random.shuffle(self.idx_list)
        
    def __len__(self):
        if self.is_training:
            return 2048 * 8
        else:
            return TEST_SIZE
    
    def __getitem__(self, index):
        if self.is_training:
            _ = index
            idx = self.idx_list[random.randrange(len(self.idx_list) - TEST_SIZE)]
        else:
            assert index < TEST_SIZE
            idx = self.idx_list[len(self.idx_list) - TEST_SIZE + index]
        
        m1 = idx % 1000
        m2 = idx // 1000

        ain = f"{m1}*{m2}="
        aout = f"{m1*m2}"
        
        ain = [0]*(self.seq_len - len(ain)) + [self.char2idx[c] for c in ain] 
        aout = [0]*(self.aout_len - len(aout)) + [self.char2idx[c] for c in aout]
        
        assert len(ain) == self.seq_len 
        assert len(aout) == self.aout_len 
        
        return torch.LongTensor(ain), torch.LongTensor(aout)


if __name__ == '__main__':
    ds = NlpM3(is_training=False)

    for i in range(len(ds)):
        ain, aout = ds[i]
        print(' | ', end="")
        for idx in ain:
            c = ds.alphabet[idx]
            print(c, end='')
        print(' | ', end="")
        for idx in aout:
            c = ds.alphabet[idx]
            print(c, end='')
        print(' | ')