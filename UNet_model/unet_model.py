# full assembly of the sub-parts to form the complete net

import torch
import torchaudio
from unet_parts import *

class UNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(513, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(641, 513, bilinear)
        self.up2 = up(512, 128, bilinear)
        self.up3 = up(256, 64, bilinear)
        self.up4 = up(128, 64, bilinear)
        self.outc = outconv(513, n_classes)

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
        # torchaudio.save('../x_0.wav', x[0].unsqueeze(0), sample_rate=8192)
        # torchaudio.save('../x_1.wav', x[1].unsqueeze(0), sample_rate=8192)
        # print("x shape:", x.shape)
        # print("x[0]", x[0].shape)
        # print("x[1]", x[1].shape)
        # X = torchaudio.transforms.Spectrogram(1024, 1024, 764, window_fn = torch.hann_window.cuda())(x)
        X = torch.stft(x, 1024, 764, 1024)
        print("pasé el torch.stft(x, 1024, 764, 1024)")
        print(X.shape)
        X1 = self.down1(X)
        print("pasé el self.inc(X)")
        print(X1.shape)
        X = self.up1(X, X1)
        print("pasé el self.up1(X, X1)")
        print(X.shape)
        X = self.outc(X)
        print("pasé el self.outx(X)")
        print(X.shape)
        X = torch.sigmoid(X)
        print("pasé el sigmoid")
        print(X.shape)
        x = torch.istft(X, 1024, 764, 1024)
        return x
