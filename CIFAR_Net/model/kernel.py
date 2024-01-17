import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class Concurrent(nn.Sequential):
    def __init__(self, axis=1, stack=False):
        super(Concurrent, self).__init__()
        self.axis = axis
        self.stack = stack  
    
    def forward(self, x):
        out = []
        for module in self._modules.values():
            out.append(module(x))
        if self.stack:
            out = torch.stack(out, dim=self.axis)
        else:
            out = torch.cat(out, dim=self.axis)
        return out

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=True, relu=True, dilation=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=dilation*(kernel_size-1)//2, dilation=dilation, bias=False)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)
        
    def forward(self, x):
        assert x.size(1) == self.inp_dim
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class SKConvBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, stride=1, num_branches=2, reduction=16, min_channels=8):
        super(SKConvBlock, self).__init__()
        self.num_branches = num_branches
        self.out_channels = out_dim
        self.stride = stride
        mid_channels = max(inp_dim // reduction, min_channels)

        self.branches = Concurrent(stack=True)
        for i in range(num_branches):
            dilation = 1 + i
            self.branches.add_module("branch{}".format(i+1), 
                                     Conv(inp_dim, out_dim, kernel_size=2*(i+1)+1, stride=self.stride, dilation=dilation))
        
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = Conv(inp_dim=out_dim, out_dim=mid_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(mid_channels, out_dim * num_branches, kernel_size=1, bias=False, stride=1)

        self.softmax = nn.Softmax(dim=1)
        self.alpha = torch.Tensor()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    
    def forward(self, x):
        y = self.branches(x)

        u = y.sum(dim=1)
        s = self.pool(u)
        z = self.fc1(s)
        w = self.fc2(z)

        batch = w.size(0)
        w = w.view(batch, self.num_branches, self.out_channels)
        w = self.softmax(w)

        self.alpha = w

        w = w.unsqueeze(-1).unsqueeze(-1)

        y = y * w
        y = y.sum(dim=1)
        return self.maxpool(y) 