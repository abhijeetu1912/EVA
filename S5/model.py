import torch
import torch.nn as nn
import torch.nn.functional as F


def Normalization(norm_type, out_channels):
    if norm_type == "BN":
        return nn.BatchNorm2d(out_channels)
    elif norm_type == "LN":
        return nn.GroupNorm(1, out_channels)
    else:
        return nn.GroupNorm(4, out_channels)


# construct cnn class
class Net(nn.Module):
    def __init__(self, norm_type):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size = 5, padding = 2)  # apply 64 channel convolution with kernel size 5*5 on image
        self.conv2 = nn.Conv2d(96, 16, kernel_size = 1)  # apply 64 channel convolution with kernel size 1*1 on image
        self.conv3 = nn.Conv2d(16, 16, kernel_size = 3)  # apply 64 channel convolution with kernel size 3*3 on image
        self.conv4 = nn.Conv2d(16, 8, kernel_size = 1)  # apply 64 channel convolution with kernel size 1*1 on image
        self.conv5 = nn.Conv2d(8, 16, kernel_size = 3)  # apply 64 channel convolution with kernel size 3*3 on image
        self.conv6 = nn.Conv2d(16, 10, kernel_size = 3)  # apply 64 channel convolution with kernel size 3*3 on image

        self.batch_norm1 = Normalization(norm_type, 96)  # normalization after convolution layer 1
        self.batch_norm2 = Normalization(norm_type, 16)  # normalization after convolution layer 2
        self.batch_norm3 = Normalization(norm_type, 16)  # normalization after convolution layer 3
        self.batch_norm4 = Normalization(norm_type, 8)  # ormalization after convolution layer 4
        self.batch_norm5 = Normalization(norm_type, 16)  # normalization after convolution layer 5

        self.dropout = nn.Dropout(0.05)

        self.gap = nn.AdaptiveAvgPool2d(1)


    def forward(self, img):
        x = F.relu(self.batch_norm1(self.conv1(img)))  # input = 1 * 28 * 28, output = 96 * 28 * 28, rf = 5*5
        x = F.relu(self.batch_norm2(self.conv2(x)))  # input = 96 * 28 * 28, output = 16 * 28 * 28, rf = 5*5
        x = F.max_pool2d(x, 2)  # input = 16 * 28 * 28, output = 16 * 14 * 14, rf = 6*6
        x = self.dropout(x)

        x = F.relu(self.batch_norm3(self.conv3(x)))  # input = 16 * 14 * 14, output = 16 * 12 * 12, rf = 10*10
        x = F.relu(self.batch_norm4(self.conv4(x)))  # input = 16 * 12 * 12, output = 8 * 12 * 12, rf = 10*10
        x = F.max_pool2d(x, 2)  # input = 8 * 12 * 12, output = 8 * 6 * 6, rf = 12*12
        x = self.dropout(x)
        
        x = F.relu(self.batch_norm5(self.conv5(x)))  # input = 8 * 6 * 6, output = 16 * 4 * 4, rf = 20*20
        x = self.conv6(x)  # input = 16 * 4 * 4, output = 10 * 2 * 2, rf = 28*28
        x = self.gap(x)  # input = 10 * 2 *2, output = 10 * 1 * 1, rf = 32*32
        out = x.view(-1, 10)  # flatten cnn embedding

        return out