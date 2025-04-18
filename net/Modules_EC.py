# -*- coding: utf-8 -*-
import torch.nn as nn

affine_par = True
import torch
from torch.nn import functional as F
from torch.autograd import Variable




class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=1)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x


class BasicDeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, out_padding=0, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicDeConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, dilation=dilation, output_padding=out_padding, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x


"""
    Position Attention Module (PAM)
"""










class ODE(nn.Module):
    def __init__(self, in_channels):
        super(ODE, self).__init__()
        self.F1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.F2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, feature_map):
        f1 = self.F1(feature_map)
        f2 = self.F2(f1 + feature_map)
        tt=torch.cat([f1, f2], dim=1)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(tt))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(tt))))
        alpha =  self.sigmoid(avg_out + max_out)
        out = feature_map + f1 * alpha + f2 * (1 - alpha)
        return out



class REU6(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(REU6, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.out_y = nn.Sequential(
            BasicConv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1),
            BasicConv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=3, padding=1)
        )

        self.out_B = nn.Sequential(
            BasicDeConv2d(in_channels, mid_channels // 2, kernel_size=3, stride=2, padding=1, out_padding=1),
            BasicConv2d(mid_channels // 2, mid_channels // 4, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels // 4, 1, kernel_size=3, padding=1)
        )
        self.ode = ODE(in_channels)

    def forward(self, x, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=x.size()[2:], mode='bilinear', align_corners=True)  # 2,1,12,12->2,1,48,48
        yt = self.conv(torch.cat([x, prior_cam.expand(-1, x.size()[1], -1, -1)], dim=1))

        ode_out = self.ode(yt)
        bound = self.out_B(ode_out)
        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1
        y = r_prior_cam.expand(-1, x.size()[1], -1, -1).mul(x)

        cat2 = torch.cat([y, ode_out], dim=1)  # 2,128,48,48
        y = self.out_y(cat2)
        y = y + prior_cam
        return y, bound




class REM11(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(REM11, self).__init__()
        self.REU_f1 = REU6(in_channels, mid_channels)
        self.REU_f2 = REU6(in_channels, mid_channels)
        self.REU_f3 = REU6(in_channels, mid_channels)
        self.REU_f4 = REU6(in_channels, mid_channels)

    def forward(self, x, prior_0, pic):
        f1, f2, f3, f4 = x

        f4_out=prior_0
        f3_out, bound_f3 = self.REU_f3(f3, f4_out)  # b,1,24,24 b,1,96,96
        f3 = F.interpolate(f3_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f3 = F.interpolate(bound_f3, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384

        f2_out, bound_f2 = self.REU_f2(f2, f3_out)  # b,1,48,48 b,1,192,192
        f2 = F.interpolate(f2_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f2 = F.interpolate(bound_f2, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384

        f1_out, bound_f1 = self.REU_f1(f1, f2_out)  # b,1,96,96 b,1,384,384
        f1 = F.interpolate(f1_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f1 = F.interpolate(bound_f1, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f4=bound_f3
        f4=f4_out

        return f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1

