from nlp_a60 import NlpA60
from tf.encoder import TfEncoder
from torch.utils.data import DataLoader
from torch import nn 
import torch.optim
import math
from argparse import ArgumentParser
parser = ArgumentParser("")
parser.add_argument("--epochs", "-e", type=int, default=1000)
parser.add_argument("--lr", type=float, default=0.2)
args = parser.parse_args()

batch_size = 32
nc = 128
lr_init = args.lr
depth = 4

train_dataset = NlpA60()
train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
         
device = "cuda:0"

class Net(nn.Module): 
    def __init__(self, nc):
        super().__init__()
        self.embed = nn.Embedding(len(train_dataset.char2idx), nc, padding_idx=None)
        self.encoder = TfEncoder(train_dataset.seq_len, nc, bmax=batch_size,
            depth=depth, device=device, grad_norm_r=2)
        self.aout_len = train_dataset.aout_len
        self.mlp = nn.Linear(nc,len(train_dataset.alphabet)-2, bias=True)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        
        x = x[:, -self.aout_len:, :]

        x = self.mlp(x)
        return x

net = Net(nc=nc).to(device)

ckpt_path = "ckpt_a60.pth"
try:
    ckpt = torch.load(ckpt_path)
    net.load_state_dict(ckpt['model_data'])
    net.encoder.load_model(ckpt['encoder'])
    print("load model from", ckpt_path)
except Exception as ex:
    print("load model error", ex)

my_loss = nn.CrossEntropyLoss()

def _get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm 

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
train_loss_min = math.inf
for epoch in range(args.epochs):
    train_loss = 0
    lr = lr_init
    for ain, aout in train_loader:
        ain = ain.to(device)
        aout = aout.to(device).view(-1)
        _zero_grad(net)
        y = net(ain)
        y = y.view(-1, y.shape[-1])

        loss = my_loss(y, aout)

        train_loss += loss.detach().item()
        assert math.isfinite(train_loss), "请降低学习率后恢复训练"

        loss.backward()

        # 自动调节学习率
        dval = _get_grad_norm(net) ** 2
        dval += net.encoder.util_grad(0)
        lr = lr_init / net.encoder.util_grad(1, dval)

        _renew_model(net, lr)
        net.encoder.renew(lr)

    print(epoch+1, f"train_loss {train_loss:.3f}, lr {lr:.3f}")
    if train_loss < train_loss_min*1.1:
        train_loss_min = min(train_loss_min, train_loss)
        torch.save(dict(model_data=net.state_dict(), encoder=net.encoder.save_model()), ckpt_path)
        print(f"save_model {train_loss_min:.3f}")

net.eval()

total = 0
err = 0
parse_s = lambda a: ("".join([train_dataset.alphabet[_] for _ in a])).strip()

test_dataset = NlpA60(is_training=False)

with torch.no_grad():
    for ain, aout in test_dataset:
        sin = parse_s(ain)
        sout = parse_s(aout)
        
        ain = ain.to(device)
        ain.unsqueeze_(0)
        pred = net(ain)
        pred = torch.argmax(pred, -1).cpu().view(-1)
        spred = parse_s(pred)

        total += 1
        
        if spred != sout: 
            print(f"{sin}\n  {spred}\n  {sout}(expected)")
            err += 1
        if total >= len(test_dataset): break


print(f"total: {total} acc: {(total-err)/total*100:.3f}% ")