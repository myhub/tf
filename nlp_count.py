import torch
import os, sys
import random
from random import SystemRandom
random = SystemRandom()
from torch.utils.data import Dataset

TEST_SIZE = 100000

class NlpCount(Dataset): 
    def __init__(self, seq_len=128, aout_len=4, is_training=True):
        self.a_z = []
        self.digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        for i in range(26):
            self.a_z += chr(i + ord('a'))

        self.BLANK = chr(2)
        self.alphabet = [self.BLANK] + self.digits + self.a_z + [' ', '+', '-', '*', '/', "="]
        self.alphabet += [
            "How ", "many ", "letters ", "are ", "there ", "in ", "the ", "following ", "string: ", "\"", "?" 
        ]

        self.seq_len = seq_len
        self.aout_len = aout_len
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
        _ = index
        rand_num = random.random()
        if rand_num < 0.05:
            ss = random.choice(self.a_z) * random.randint(1, 99)
        elif rand_num < 0.25:
            ss = random.choices(self.a_z, k=random.randint(1, 99))
        else:
            ss = random.choices(self.a_z + self.digits, k=random.randint(1, 99))

        ain = ["How ", "many ", "letters ", "are ", "there ", "in ", "the ", "following ", "string: ", "\""]
        ain += ss            
        ain += ["\"", "?"] + [self.BLANK]*self.aout_len
        aout = 0
        for s in ss:
            if str(s).isalpha(): aout += 1
        aout = str(aout)
        
        assert len(ain) <= self.seq_len

        ain = [self.char2idx[_] for _ in ain]
        aout = [self.char2idx[_] for _ in aout]
        len_aout = len(aout)
        ain = [0]*(self.seq_len - len(ain)) + ain
        aout = [0]*(self.aout_len - len(aout)) + aout
        
        return torch.LongTensor(ain), torch.LongTensor(aout)


if __name__ == '__main__':
    ds = NlpCount()

    for i in range(5000):
        ain, aout = ds[i]
        print(ain.shape, aout.shape)

        for idx in ain:
            c = ds.alphabet[idx]
            if c == ds.BLANK: continue
            print(ds.alphabet[idx], end='')
        print()
        for idx in aout:
            c = ds.alphabet[idx]
            if c == ds.BLANK: continue
            print(ds.alphabet[idx], end='')
        print()
