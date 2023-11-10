import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
     Each convolutional layer is followed by a BatchNorm layer and a ReLU layer.
    """
    def __init__(self, cin, cout):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(cout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU6(out)
        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, cin, cout):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(cin, cin, kernel_size=3, padding=1, groups=cin)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pointwise = nn.Conv2d(cin, cout, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn1(out)
        out = nn.ReLU6(out)
        out = self.pointwise(out)
        out = self.bn2(out)
        out = nn.ReLU6(out)
        return out


class DBlock(nn.Module):   # bottleneck?
    def __init__(self, cin, cout):
        super(DBlock, self).__init__()
        self.conv_residual = ConvBlock(cin, cout)

        self.conv_dw1 = DepthwiseSeparableConv(cin, cout)   # not sure about cin
        self.conv1 = ConvBlock(cout, cout)
        self.conv_dw2 = DepthwiseSeparableConv(cout, cout)
        self.conv2 = nn.Conv2d(cout, cout, stride=1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        residual = self.conv_residual(x)

        out = self.conv_dw1(x)
        out = self.conv1(out)
        out = nn.ReLU6(out)
        out = self.conv_dw2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        out = nn.ReLU6(out)
        return out


class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride=1, expand_ratio=1):    # expand_ratio: dim of hidden layer
        super(InvertedResidual, self).__init__()
        self.hidden_dim = cin * expand_ratio
        self.stride = stride
        self.conv1 = ConvBlock(cin, self.hidden_dim)
        self.conv_dw1 = DepthwiseSeparableConv(self.hidden_dim, self.hidden_dim)
        self.conv2 = nn.Conv2d(self.hidden_dim, cout, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        out = self.conv_dw1(x)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride == 1:
            residual = x
            out += residual

        out = nn.ReLU6(out)
        return out


class PortraitNet(nn.Module):
    def __init__(self):
        super(PortraitNet, self).__init__()
        self.first_conv = nn.Conv2d(3, 32, kernel_size=2, stride=2)    # kernel size = 2 or 3 ?

        # Encoder
        # stage 1  (112, 112)
        self.conv1 = InvertedResidual(32, 16, stride=1, expand_ratio=1)

        self.conv2 = InvertedResidual(16, 24, stride=2, expand_ratio=6)
        self.conv3 = InvertedResidual(24, 24, stride=1, expand_ratio=6)
        self.stage1 = nn.Sequential(self.conv1, self.conv2, self.conv3)

        # stage 2  (56, 56)
        self.conv4 = InvertedResidual(24, 32, stride=2, expand_ratio=6)
        self.conv5 = InvertedResidual(32, 32, stride=1, expand_ratio=6)
        self.conv6 = InvertedResidual(32, 32, stride=1, expand_ratio=6)
        self.stage2 = nn.Sequential(self.conv4, self.conv5, self.conv6)

        # stage 3  (28, 28)
        self.conv7 = InvertedResidual(32, 64, stride=2, expand_ratio=6)
        self.conv8 = InvertedResidual(64, 64, stride=1, expand_ratio=6)
        self.conv9 = InvertedResidual(64, 64, stride=1, expand_ratio=6)
        self.conv10 = InvertedResidual(64, 64, stride=1, expand_ratio=6)
        self.stage3 = nn.Sequential(self.conv7, self.conv8, self.conv9, self.conv10)

        # stage 4  (14, 14)
        self.conv11 = InvertedResidual(64, 96, stride=1, expand_ratio=6)
        self.conv12 = InvertedResidual(96, 96, stride=1, expand_ratio=6)
        self.conv13 = InvertedResidual(96, 96, stride=1, expand_ratio=6)

        self.conv14 = InvertedResidual(96, 160, stride=2, expand_ratio=6)
        self.conv15 = InvertedResidual(160, 160, stride=1, expand_ratio=6)
        self.conv16 = InvertedResidual(160, 160, stride=1, expand_ratio=6)
        self.stage4 = nn.Sequential(self.conv11, self.conv12, self.conv13, self.conv14, self.conv15, self.conv16)

        # stage 5 (7, 7)
        self.conv17 = InvertedResidual(160, 320, stride=1, expand_ratio=6)
        self.stage5 = nn.Sequential(self.conv17)

        # UpSampler
        self.upsample1 = nn.ConvTranspose2d(96, 96, kernel_size=4, stride=2, padding=1, bias=False)     # padding
        self.upsample2 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample3 = nn.ConvTranspose2d(24, 24, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample4 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample5 = nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1, bias=False)

        # Decoder
        self.dblock1 = DBlock(320, 96)
        self.dblock3 = DBlock(96, 32)
        self.dblock4 = DBlock(32, 24)
        self.dblock5 = DBlock(24, 16)
        self.dblock6 = DBlock(16, 8)

        # Mask
        self.mask = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)

        # Boundary
        self.boundary = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        encoder_out = self.first_conv(x)
        encoder_out1 = self.stage1(encoder_out)
        encoder_out2 = self.stage2(encoder_out1)
        encoder_out3 = self.stage3(encoder_out2)
        encoder_out4 = self.stage4(encoder_out3)
        encoder_out5 = self.stage5(encoder_out4)

        # Decoder
        decoder_out1 = self.upsample1(self.dblock1(encoder_out5))  # (14, 14, 96)
        decoder_out2 = self.upsample2(self.dblock3(decoder_out1 + encoder_out4))  # (28, 28, 64)
        decoder_out3 = self.upsample3(self.dblock4(decoder_out2 + encoder_out3))  # (56, 56, 32)
        decoder_out4 = self.upsample4(self.dblock5(decoder_out3 + encoder_out2))  # (112, 112, 24)
        decoder_out5 = self.upsample5(self.dblock6(decoder_out4 + encoder_out1))  # (224, 224, 16)

        mask_out = self.mask(decoder_out5)
        boundary_out = self.boundary(decoder_out5)

        return mask_out, boundary_out









