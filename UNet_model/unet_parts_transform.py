# subparts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
class down(nn.Module):
    '''(conv => BN => LeackyReLU)'''
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class up(nn.Module):
    '''
        (concatenation => deconv => BN => ReLU => Dropout )
        as proposed in SINGING VOICE SEPARATION WITH DEEP
        U-NET CONVOLUTIONAL NETWORK
    '''
    def __init__(self, in_ch, out_ch, kernel_size, stride, output_padding, index):
        super(up, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, output_padding=output_padding)
        if index > 3:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 1, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
        else:
            # 50% dropout for the first 3 layers
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 1, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
#                nn.Dropout(p=0.5)
            )

    def forward(self, x1, x2):
        print("x_antes de up_conv:", x2.shape)
        x = self.up_conv(x2)
        print("x_dp de up_conv:", x.shape)
        x = torch.cat([x, x1], dim=1)
        print("dp de cat", x.shape)
        x = self.deconv(x)


        return x

class last_layer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, output_padding):
        super(last_layer, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, output_padding=output_padding),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.deconv(x)

        return x

