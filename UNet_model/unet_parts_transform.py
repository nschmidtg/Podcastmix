# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


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
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(up, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(17, out_ch, kernel_size, stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

    def forward(self, x1, x2):
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        print("x1:", x1.shape)
        print("x2:", x2.shape)
        x = torch.cat([x2, x1], dim=1)
        print("x_ before deconv:", x.shape)
        x = self.deconv(x)
        print("x_after devonc:", x.shape)
        return x

# class down(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(down, self).__init__()
#         self.mpconv = nn.Sequential(
#             nn.MaxPool2d(2),
#             double_conv(in_ch, out_ch)
#         )

#     def forward(self, x):
#         x = self.mpconv(x)
#         return x


# class up(nn.Module):
#     def __init__(self, in_ch, out_ch, bilinear=False):
#         super(up, self).__init__()

#         #  would be a nice idea if the upsampling could be learned too,
#         #  but my machine do not have enough memory to handle all those weights
#         if bilinear:
#             #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.bilinear = True
#         if not bilinear:
#             self.bilinear = False
#             self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

#         self.conv = double_conv(in_ch, out_ch)

#     def forward(self, x1, x2):
#         if self.bilinear:
#             x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)
#         else:
#             x1 = self.up(x1)
        
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
#                         diffY // 2, diffY - diffY//2))

#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
