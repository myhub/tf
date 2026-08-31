import torch
import torch.autograd
import torch.nn
import os
from pathlib import Path
import random

from encoder import TfEncoder

if __name__ == "__main__":
    device = torch.device("cuda", 0)
    bmax = 32
    t = 1024
    c = 512

    if 1:
        net = TfEncoder(t, c, bmax, device=device)
        print(net)

        for i in range(200):
            b = random.randint(1, 4)
            x = torch.randn((b, t, c), device=device, requires_grad=True)
            y = net(x)
            loss = (y**2).mean() * 1.123
            if (i % 5 == 0):
                print("loss", loss.detach().item())
            loss.backward()
            r1 = net.util_grad(0)
            r2 = net.util_grad(1, r1)
            net.renew(0.1 / r2)
        model_data = net.save_model()
        
        del net

    net = TfEncoder(t, c, bmax, device=device)
    print("load_model",
        net.load_model(model_data)
    )

    for i in range(20):
        b = random.randint(1, 4)
        x = torch.randn((b, t, c), device=device, requires_grad=True)
        y = net(x)
        loss = (y**2).mean()
        if (i % 1 == 0):
            print("loss", loss.detach().item())
        loss.backward()
        net.renew(0.01)

   