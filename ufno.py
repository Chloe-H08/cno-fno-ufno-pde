import torch
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial

torch.manual_seed(0)

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights)
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.down = nn.Sequential(
            nn.Conv3d(input_channels, output_channels * 2, kernel_size=kernel_size, stride=2, padding=1),
            nn.GroupNorm(1, output_channels * 2),
            nn.ReLU(),
            nn.Conv3d(output_channels * 2, output_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x_down = self.down(x)
        x_up = self.up(x_down)
        return x_up


class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width=16):  # was 12
        super(SimpleBlock3d, self).__init__()
        self.width = width
        self.fc0 = nn.Linear(12, width)

        self.conv0 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv1 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv2 = SpectralConv3d(width, width, modes1, modes2, modes3)

        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)

        self.unet = U_net(width, width, kernel_size=3, dropout_rate=0)

        self.fc1 = nn.Linear(width, 64)   # was 32
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        b, x_len, y_len, z_len, _ = x.shape
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(b, self.width, -1)).view(b, self.width, x_len, y_len, z_len)
        x = F.relu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(b, self.width, -1)).view(b, self.width, x_len, y_len, z_len)
        x = F.relu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(b, self.width, -1)).view(b, self.width, x_len, y_len, z_len)
        x3 = self.unet(x)
        x = F.relu(x1 + x2 + x3)

        x = x.permute(0, 2, 3, 4, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Net3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(Net3d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock3d(modes1, modes2, modes3, width)


    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        x = F.pad(F.pad(x, (0,0,0,8,0,8), "replicate"), (0,0,0,0,0,0,0,8), 'constant', 0)
        x = self.conv1(x)
        x = x.view(batchsize, size_x+8, size_y+8, size_z+8, 1)[..., :-8,:-8,:-8, :]
        return x.squeeze()


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c