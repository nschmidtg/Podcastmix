# full assembly of the sub-parts to form the complete net
import math
import torch
import torchaudio
from unet_parts import *

class UNet(torch.nn.Module):
    #def __init__(self, n_channels, n_classes, bilinear=True):
    def __init__(self, segment, sample_rate, fft_size, hop_size, window_size, kernel_size_c, stride_c, kernel_size_d, stride_d):
        super(UNet, self).__init__()
        # self.inc = inconv(n_channels, 64)
        # self.down1 = down(513, 128)
        # self.down2 = down(128, 256)
        # self.down3 = down(256, 512)
        # self.down4 = down(512, 512)
        # self.up1 = up(641, 513, bilinear)
        # self.up2 = up(512, 128, bilinear)
        # self.up3 = up(256, 64, bilinear)
        # self.up4 = up(128, 64, bilinear)
        # self.outc = outconv(513, n_classes)
        self.segment = segment
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.kernel_size_c = kernel_size_c
        self.stride_c = stride_c
        self.kernel_size_d = kernel_size_d
        self.stride_d = stride_d

        self.window = torch.hamming_window(self.window_size).cuda()
        self.number_of_samples_in_x = segment * sample_rate
        self.input_number_frames = math.floor(self.number_of_samples_in_x / hop_size) + 1

        self.down1 = down(1, 16, self.kernel_size_c, self.stride_c)
        self.up1 = up(16, 1, self.kernel_size_d, self.stride_d)

    def forward(self, x):
        # torchaudio.save('../x_0.wav', x[0].unsqueeze(0), sample_rate=8192)
        # torchaudio.save('../x_1.wav', x[1].unsqueeze(0), sample_rate=8192)
        # print("x shape:", x.shape)
        # print("x[0]", x[0].shape)
        # print("x[1]", x[1].shape)
        # X = torchaudio.transforms.Spectrogram(1024, 1024, 764, window_fn = torch.hann_window.cuda())(x)
        # usar hamming window: default for speech
        X = torch.stft(
            x,
            self.fft_size,
            self.hop_size,
            self.window_size,
            return_complex=True,
            window=self.window
        )
        X = torch.abs(X)
        # normalize
        X -= torch.min(X)
        X /= torch.max(X)
        print("pasé el torch.stft")
        # add single channel dimension after batch dimension
        X = X.unsqueeze(1)
        print(X.shape)
        X1 = self.down1(X)
        print("pasé el self.down1(X)")
        print("X:", X1.shape)
        print("X1:", X.shape)
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
