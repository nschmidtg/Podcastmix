# full assembly of the sub-parts to form the complete net

import torch

from unet_parts import *

class UNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, bilinear)
        self.up2 = up(512, 128, bilinear)
        self.up3 = up(256, 64, bilinear)
        self.up4 = up(128, 64, bilinear)
        self.outc = outconv(64, n_classes)

    # def forward(self, x):
    #     X = torch.stft(x, 1024, 764, 1024)
    #     X1 = self.inc(X)
    #     X2 = self.down1(X1)
    #     X3 = self.down2(X2)
    #     X4 = self.down3(X3)
    #     X5 = self.down4(X4)
    #     X = self.up1(X5, X4)
    #     X = self.up2(X, X3)
    #     X = self.up3(X, X2)
    #     X = self.up4(X, X1)
    #     X = self.outc(X)
    #     X = torch.sigmoid(X)
    #     x = torch.istft(X, 1024, 764, 1024)
    #     return x

    def forward(self, x):
        X = torch.stft(x, 1024, 764, 1024)
        print("pasé el torch.stft(x, 1024, 764, 1024)")
        print(X.shape)
        X1 = self.inc(X)
        print("pasé el self.inc(X)")
        X = self.up1(X, X1)
        print("pasé el self.up1(X, X1)")
        X = self.outc(X)
        print("pasé el self.outx(X)")
        X = torch.sigmoid(X)
        print("pasé el sigmoid")
        x = torch.istft(X, 1024, 764, 1024)
        return x
