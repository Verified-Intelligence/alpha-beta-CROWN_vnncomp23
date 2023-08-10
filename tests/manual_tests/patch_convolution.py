import sys
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import models


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, bn=True, kernel=3):
        super(BasicBlock, self).__init__()
        self.bn = bn
        if kernel == 3:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=(not self.bn))
        elif kernel == 2:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=2, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=2,
                                   stride=1, padding=0, bias=(not self.bn))
        elif kernel == 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                                   stride=1, padding=0, bias=(not self.bn))
        else:
            exit("kernel not supported!")

        if self.bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if self.bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=(not self.bn)),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=(not self.bn)),
                )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks=2, num_classes=10, in_planes=64, bn=True, last_layer="avg"):
        super().__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.last_layer = last_layer
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=2, padding=1, bias=not self.bn)
        if self.bn: self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        if self.last_layer == "avg":
            self.avg2d = nn.AvgPool2d(4)
            self.linear = nn.Linear(in_planes * 2 * block.expansion, num_classes)
        elif self.last_layer == "dense":
            self.linear1 = nn.Linear(in_planes * 2 * block.expansion * 16, 100)
            self.linear2 = nn.Linear(100, num_classes)
        else:
            exit("last_layer type not supported!")

    def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        if self.last_layer == "avg":
            out = self.avg2d(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif self.last_layer == "dense":
            out = out.view(out.size(0), -1)
            out = F.relu(self.linear1(out))
            out = self.linear2(out)
        return out


def cifar_conv_big():
    model = nn.Sequential(
        nn.BatchNorm2d(3),
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model

def cifar_conv_small():
    model = nn.Sequential(
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.Conv2d(3, 2, 3, stride=1, padding=1),
        # nn.ReLU(),
        nn.Conv2d(2, 2, 4, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1922,10),
    )
    return model

class cnn_4layer_test(nn.Module):
    def __init__(self):
        super(cnn_4layer_test, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(3)
        self.shortcut = nn.Conv2d(3, 3, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(3, 3, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(192, 10)

    def forward(self, x):
        x_ = x
        x = F.relu(self.conv1(self.bn(x)))
        x += self.shortcut(x_)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x
# This auto_LiRPA release supports a new mode for computing bounds for
# convolutional layers. The new "patches" mode implementation makes full
# backward bounds (CROWN) for convolutional layers significantly faster by
# using more efficient GPU operators, but it is currently stil under beta test
# and may not support any architecutre.  The convolution mode can be set by
# the "conv_mode" key in the bound_opts parameter when constructing your
# BoundeModule object.  In this test we show the difference between Patches
# mode and Matrix mode in memory consumption.

device = 'cpu'
conv_mode = sys.argv[1] if len(sys.argv) > 1 else 'patches' # conv_mode can be set as 'matrix' or 'patches'

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

## Step 1: Define the model
# model_ori = ResNet(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="dense")
# model_ori = ResNet(BasicBlock, num_blocks=2, in_planes=8, bn=True, last_layer="dense")
# model_ori = models.model_resnet(width=1, mult=4)
# model_ori = models.ResNet18(in_planes=2)
# model_ori.load_state_dict(torch.load("data/cifar_base_kw.pth")['state_dict'][0])
model_ori = cifar_conv_small()
conv_cnt = 0
for m in model_ori.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.running_mean.data.copy_(torch.randn_like(m.running_mean))
        m.running_var.data.copy_(torch.abs(torch.randn_like(m.running_var)))

model_ori = model_ori.to(device=device)

## Step 2: Prepare dataset as usual
# test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())

normalize = torchvision.transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True,
                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize]))
# For illustration we only use 1 image from dataset
N = 1
n_classes = 10

image = torch.Tensor(test_data.data[:N]).reshape(N,3,32,32)
# Convert to float
image = image.to(torch.float32) / 255.0
if device == 'cuda':
    image = image.cuda()

## Step 3: wrap model with auto_LiRPA
# The second parameter is for constructing the trace of the computational graph, and its content is not important.
# The new "patches" conv_mode provides an more efficient implementation for convolutional neural networks.
model = BoundedModule(model_ori, image, bound_opts={"conv_mode": conv_mode}, device=device)

## Step 4: Compute bounds using LiRPA given a perturbation
eps = 0.1
norm = np.inf
ptb = PerturbationLpNorm(norm = norm, eps = eps)
image = BoundedTensor(image, ptb)
# Get model prediction as usual
pred = model(image)

# Compute bounds
torch.cuda.empty_cache()
print('Using {} mode to compute convolution.'.format(conv_mode))
lb, ub = model.compute_bounds(IBP=False, C=None, method='backward')

## Step 5: Final output
# pred = pred.detach().cpu().numpy()
lb = lb.detach().cpu().numpy()
ub = ub.detach().cpu().numpy()
for i in range(N):
    # print("Image {} top-1 prediction {}".format(i, label[i]))
    for j in range(n_classes):
        print("f_{j}(x_0): {l:8.5f} <= f_{j}(x_0+delta) <= {u:8.5f}".format(j=j, l=lb[i][j], u=ub[i][j]))
    print()

# Print the GPU memory usage
# print('Memory usage in "{}" mode:'.format(conv_mode))
# print(torch.cuda.memory_summary())
