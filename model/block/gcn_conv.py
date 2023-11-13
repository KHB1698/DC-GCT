import torch
import torch.nn as nn
from timm.models.layers import DropPath


class Gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A


class Gcn_block(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0.05,
                 residual=True):

        super().__init__()
        self.inplace = True

        self.momentum = 0.1
        self.gcn = Gcn(in_channels, out_channels, kernel_size)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels, momentum=self.momentum),
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(0.05),
            nn.Conv2d(
                out_channels,
                out_channels,
                (1, 1),
                (stride, 1),
                padding = 0,
            ),
            nn.BatchNorm2d(out_channels, momentum=self.momentum),
            nn.Dropout(dropout, inplace=self.inplace),


        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels, momentum=self.momentum),
            )

        # self.relu = nn.ReLU(inplace=self.inplace)
        self.gelu = nn.GELU()

    def forward(self, x, A):
        N,J,C = x.shape
        x = x.permute(0,2,1).contiguous().view(N,C,1,J)
        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res
        x = x.view(N,-1,J).permute(0,2,1).contiguous()

        return self.gelu(x),A
