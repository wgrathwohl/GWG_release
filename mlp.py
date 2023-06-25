import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x * torch.sigmoid(x)


def mlp_ebm(nin, nint=256, nout=1):
    return nn.Sequential(
        nn.Linear(nin, nint),
        Swish(),
        nn.Linear(nint, nint),
        Swish(),
        nn.Linear(nint, nint),
        Swish(),
        nn.Linear(nint, nout),
    )


class MLPEBM_cat(nn.Module):
    def __init__(self, nin, n_proj, n_cat=256, nint=256, nout=1):
        super().__init__()
        self.proj = nn.Linear(n_cat, n_proj)
        self.n_proj = n_proj
        self.net = mlp_ebm(nin * n_proj, nint, nout=nout)

    def forward(self, x):
        # xr = x.view(x.size(0) * x.size(1), x.size(2))
        # xr_p = self.proj(xr)
        xr_p = self.proj(x)
        # x_p = xr_p.view(x.size(0), x.size(1), self.n_proj)
        # x_p = x_p.view(x.size(0), x.size(1) * self.n_proj)
        x_p = xr_p.view(x.size(0), x.size(1) * self.n_proj)
        return self.net(x_p)


def conv_transpose_3x3(in_planes, out_planes, stride=1):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=3, stride=stride, padding=1, output_padding=1, bias=True)


def conv3x3(in_planes, out_planes, stride=1):
    if stride < 0:
        return conv_transpose_3x3(in_planes, out_planes, stride=-stride)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, out_nonlin=True):
        super(BasicBlock, self).__init__()
        self.nonlin1 = Swish()
        self.nonlin2 = Swish()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.out_nonlin = out_nonlin

        self.shortcut_conv = None
        if stride != 1 or in_planes != self.expansion * planes:
            if stride < 0:
                self.shortcut_conv = nn.ConvTranspose2d(in_planes, self.expansion*planes,
                                                        kernel_size=1, stride=-stride,
                                                        output_padding=1, bias=True)
            else:
                self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes,
                                               kernel_size=1, stride=stride, bias=True)


    def forward(self, x):
        out = self.nonlin1(self.conv1(x))
        out = self.conv2(out)
        if self.shortcut_conv is not None:
            out_sc = self.shortcut_conv(x)
            out += out_sc
        else:
            out += x
        if self.out_nonlin:
            out = self.nonlin2(out)
        return out


class ResNetEBM(nn.Module):
    def __init__(self, n_channels=64):
        super().__init__()
        self.proj = nn.Conv2d(1, n_channels, 3, 1, 1)
        downsample = [
            BasicBlock(n_channels, n_channels, 2),
            BasicBlock(n_channels, n_channels, 2)
        ]
        main = [BasicBlock(n_channels, n_channels, 1) for _ in range(6)]
        all = downsample + main
        self.net = nn.Sequential(*all)
        self.energy_linear = nn.Linear(n_channels, 1)

    def forward(self, input):
        input = input.view(input.size(0), 1, 28, 28)
        input = self.proj(input)
        out = self.net(input)
        out = out.view(out.size(0), out.size(1), -1).mean(-1)
        return self.energy_linear(out).squeeze()


class MNISTConvNet(nn.Module):
    def __init__(self, nc=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, nc, 3, 1, 1),
            Swish(),
            nn.Conv2d(nc, nc * 2, 4, 2, 1),
            Swish(),
            nn.Conv2d(nc * 2, nc * 2, 3, 1, 1),
            Swish(),
            nn.Conv2d(nc * 2, nc * 4, 4, 2, 1),
            Swish(),
            nn.Conv2d(nc * 4, nc * 4, 3, 1, 1),
            Swish(),
            nn.Conv2d(nc * 4, nc * 8, 4, 2, 1),
            Swish(),
            nn.Conv2d(nc * 8, nc * 8, 3, 1, 0),
            Swish(),
        )
        self.out = nn.Linear(nc * 8, 1)

    def forward(self, input):
        input = input.view(input.size(0), 1, 28, 28)
        out = self.net(input)
        out = out.squeeze()
        return self.out(out).squeeze()



class ResNetEBM_cat(nn.Module):
    def __init__(self, shape, n_proj, n_cat=256, n_channels=64):
        super().__init__()
        self.shape = shape
        self.n_cat = n_cat
        self.proj = nn.Conv2d(n_cat, n_proj, 1, 1, 0)
        self.proj2 = nn.Conv2d(n_proj, n_channels, 3, 1, 1)
        downsample = [
            BasicBlock(n_channels, n_channels, 2),
            BasicBlock(n_channels, n_channels, 2)
        ]
        main = [BasicBlock(n_channels, n_channels, 1) for _ in range(6)]
        all = downsample + main
        self.net = nn.Sequential(*all)
        self.energy_linear = nn.Linear(n_channels, 1)

    def forward(self, input):
        input = input.view(input.size(0), self.shape[1], self.shape[2], self.n_cat).permute(0, 3, 1, 2)
        input = self.proj(input)
        input = self.proj2(input)
        out = self.net(input)
        out = out.view(out.size(0), out.size(1), -1).mean(-1)
        return self.energy_linear(out).squeeze()