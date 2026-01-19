from nlp_mnist import NlpMnist
from tf.encoder import TfEncoder
from torch.utils.data import DataLoader
from torch import nn 
import torch.optim

batch_size = 32
nc = 256
lr_init = 0.15
depth = 3

train_dataset = NlpMnist()
train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
         
device = "cuda:0"

class Net(nn.Module): 
    def __init__(self, nc):
        super().__init__()
        self.embed = nn.Embedding(len(train_dataset.char2idx), nc, padding_idx=0)
        self.encoder = TfEncoder(train_dataset.seq_len, nc, bmax=batch_size,
            depth=depth, device=device)

        self.mlp = nn.Linear(nc, 10, bias=True)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        x = self.mlp(x)
        return x

net = Net(nc=nc).to(device)
my_loss = nn.CrossEntropyLoss()

def _renew_model(model, lr):
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None: continue
            p.data -= lr * p.grad.data

def _zero_grad(model):
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None: continue
            p.grad.zero_()

net.train()
epochs = 200
for epoch in range(epochs):
    train_loss = 0
    lr = lr_init
    for ain, aout in train_loader:
        ain = ain.to(device)
        aout = aout.to(device).view(-1)
        _zero_grad(net)
        y = net(ain)

        loss = my_loss(y, aout)

        train_loss += loss.detach().item()
        loss.backward()

        _renew_model(net, lr)
        net.encoder.renew(lr)

    print(epoch+1, f"train_loss {train_loss:.5f}, lr {lr:.3f}")

net.eval()

total = 0
err = 0
with torch.no_grad():
    for ain, aout in train_dataset:
        ain = ain.to(device)
        ain.unsqueeze_(0)
        aout = aout.item()
        pred = net(ain)
        pred = torch.argmax(pred, -1).item()

        total += 1
        
        if pred != aout: 
            for idx in ain[0]:
                idx = int(idx)
                c = train_dataset.alphabet[idx]
                if c == train_dataset.BLANK: continue
                print(train_dataset.alphabet[idx], end='')
            print(f" {pred} != {aout}(expected)")
            err += 1
        if total >= 1000: break


print(f"acc {(total-err)/total*100:.1f}%")