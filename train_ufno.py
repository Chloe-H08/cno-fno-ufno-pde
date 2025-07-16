import torch
import torch.nn as nn
import numpy as np
from ufno import *
from lploss import *
from ufno import Net3d 

torch.manual_seed(0)
np.random.seed(0)

import sys
import torch.nn.functional as F
sys.path.append("C:/Users/Chloe/Desktop/garcia")
from neuraloperator.neuralop.data.datasets import load_mini_burgers_1dtime


def _resize_batch(x,y):
    print (x.shape, y.shape)
    if x.ndim == 2:
        x = x.unsqueeze(1)  # [B, 1, H, W]
    if x.ndim == 3:
        x = x.unsqueeze(1)  # [B, 1, H, W]
    if y.ndim == 3:
        y = y.unsqueeze(1)  # [B, 1, H, W]
    print (x.shape, y.shape)

    x_resized = F.interpolate(x, size=(200, 24), mode='bilinear', align_corners=False)  # [B, C=1, X, T]
    y_resized = F.interpolate(y, size=(200, 24), mode='bilinear', align_corners=False)

    x_resized = x_resized.unsqueeze(3)  # insert dim at pos 3
    y_resized = y_resized.unsqueeze(3)

    # Permute to [B, X, Y, T, C]
    x_resized = x_resized.permute(0, 2, 3, 4, 1)
    y_resized = y_resized.permute(0, 2, 3, 4, 1)

    # Now add dummy channels to input to have 12 channels
    # For demonstration, replicate the single channel 12 times
    x_resized = x_resized.repeat(1, 1, 96, 1, 12)  # [B, X, 96, T, 12]
    y_resized = y_resized.repeat(1, 1, 96, 1, 1)

    # Output y stays with 1 channel: [B, X, Y, T, 1]
    print (x_resized.shape, y_resized.shape)
    return x_resized.float(), y_resized.float()

# train_a = torch.load('ufno/data/darcy_train_128.pt')['x'].to(torch.float32)[:1000]
# train_u = torch.load('ufno/data/darcy_train_128.pt')['y'].to(torch.float32)[:1000]
train_a = torch.load('ufno/data/burgers_train_16.pt')['x'].to(torch.float32)[:1000]
train_u = torch.load('ufno/data/burgers_train_16.pt')['y'].to(torch.float32)[:1000]
train_a, train_u = _resize_batch(train_a, train_u)

mode1 = 10
mode2 = 10
mode3 = 10
width = 36
device = torch.device('cuda:0')
model = Net3d(modes1=10, modes2=10, modes3=10, width=16)

# model = Net3d(mode1, mode2, mode3, width)
    
model.to(device)

print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


epochs = 1
e_start = 0
learning_rate = 0.001
scheduler_step = 2
scheduler_gamma = 0.9

batch_size = 16

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
myloss = LpLoss(size_average=False)

train_l2 = 0.0
for ep in range(1, epochs+1):
    print (ep)
    model.train()
    train_l2 = 0
    counter = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = myloss(pred.view(y.shape), y)  # Ensure shapes match
        loss.backward()
        optimizer.step()

        train_l2 += loss.item()
        counter += 1

        if counter % 100 == 0:
            print(f'epoch: {ep}, batch: {counter}/{len(train_loader)}, train loss: {loss.item()/batch_size:.4f}')
    
    scheduler.step()
    print(f'epoch: {ep}, train loss: {train_l2/train_a.shape[0]:.4f}')

    lr_ = optimizer.param_groups[0]['lr']
    if ep % 2 == 0:
        PATH = f'TrainedModels/Darcy_UFNO_{ep}ep_{width}width_{mode1}m1_{mode2}m2_{train_a.shape[0]}train_{lr_:.2e}lr'
        torch.save(model, PATH)

# test_a = torch.load('ufno/data/darcy_test_128.pt')['x'].to(torch.float32)[:400]
# test_u = torch.load('ufno/data/darcy_test_128.pt')['y'].to(torch.float32)[:400]
test_a = torch.load('ufno/data/burgers_test_16.pt')['x'].to(torch.float32)[:200]
test_u = torch.load('ufno/data/burgers_test_16.pt')['y'].to(torch.float32)[:200]
test_a, test_u = _resize_batch(test_a, test_u)

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=True)

model.eval()
loss_fn = nn.MSELoss()
test_loss = 0.0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.cuda(), yb.cuda()
        pred = model(xb)
        loss = loss_fn(pred, yb.squeeze())
        test_loss += loss.item()
print(f"Test MSE Loss: {test_loss / len(test_loader):.6f}")
