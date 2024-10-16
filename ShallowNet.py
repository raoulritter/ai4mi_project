#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec, Jose Dolz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch.nn as nn


def convBatch(
    nin,
    nout,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False,
    layer=nn.Conv2d,
    dilation=1,
):
    return nn.Sequential(
        layer(
            nin,
            nout,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            dilation=dilation,
        ),
        nn.BatchNorm2d(nout),
        nn.PReLU(),
    )


class shallowCNN(nn.Module):
    def __init__(self, nin, nout, nG=4):
        super(shallowCNN, self).__init__()
        self.conv0 = convBatch(nin, nG * 4)
        self.conv1 = convBatch(nG * 4, nG * 4)
        self.conv2 = convBatch(nG * 4, nout)

    def forward(self, input):
        x0 = self.conv0(input)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)

        return x2

    def init_weights(self, *args, **kwargs):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
