import torch
import os
import random
from torch.utils.data import Dataset
import sys
from io import StringIO

TEST_SIZE = 100000

class NlpReverse(Dataset): 
    def __init__(self, seq_len=60, align_left=True, is_training=True):
        
        self.alphabet = ['*', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        self.seq_len = seq_len
        self.aout_len = seq_len // 2

        self.is_training = is_training
        self.align_left = align_left
        
        self.char2idx = {}
        for i, c in enumerate(self.alphabet):
            self.char2idx[c] = i

    def __len__(self):
        if self.is_training:
            return 2048 * 8
        else:
            return TEST_SIZE
    
    def __getitem__(self, index):
        _ = index
        
        n1 = random.randint(1, self.aout_len)
        m1 = random.randrange(10**(n1-1), 10**n1)

        ain = str(m1)
        aout = ain[::-1]

        ain += '*'*self.aout_len

        ain = [0]*(self.seq_len - len(ain)) + [self.char2idx[c] for c in ain] 
        if self.align_left:
            aout = [self.char2idx[c] for c in aout] + [0]*(self.aout_len - len(aout))
        else:
            aout = [0]*(self.aout_len - len(aout)) + [self.char2idx[c] for c in aout]
            
        assert len(ain) == self.seq_len 
        assert len(aout) == self.aout_len 
        
        return torch.LongTensor(ain), torch.LongTensor(aout)


if __name__ == '__main__':
    ds = NlpReverse(align_left=False, seq_len=20)

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