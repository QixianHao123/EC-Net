import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log
from net.pvtv2 import pvt_v2_b2
from net.Modules_EC import *
import numpy as np
from net.layer import MultiSpectralAttentionLayer

# from net.Modules_EC import *
class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ED(nn.Module):
    def __init__(self):
        super(ED, self).__init__()
        self.reduce1 = Conv1x1(64, 64)
        self.reduce2 = Conv1x1(128, 128)
        self.reduce3 = Conv1x1(320, 320)
        self.reduce4 = Conv1x1(512, 512)

        self.block = nn.Sequential(
            ConvBNR(512+320+128+64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4,x3,x2, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x2 = self.reduce2(x2)
        x3 = self.reduce3(x3)
        x4 = self.reduce4(x4)

        x2 = F.interpolate(x2, size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size, mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4,x3,x2, x1), dim=1)
        out = self.block(out)

        return out


class EFM(nn.Module):
    def __init__(self, channel):
        super(EFM, self).__init__()
        self.conv2d = ConvBNR(channel, channel, 3)
        reduction = 16
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        self.att = MultiSpectralAttentionLayer(channel, c2wh[channel], c2wh[channel], reduction=reduction,
                                               freq_sel_method='top16')
    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c
        x = self.conv2d(x)
        x = self.att(x)
        return x


class MSCA(nn.Module):
    def __init__(self, channels=64, r=2):
        super(MSCA, self).__init__()
        out_channels = int(channels//r)
        #local_att
        self.local_att =nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding= 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(channels)
        )

        #global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        xl=self.local_att(x)

        xg = self.global_att(x)
        xlg = xg+xl
        wei = self.sig(xlg)

        return wei

def cus_sample(feat, **kwargs):
    """
    :param feat: 输入特征
    :param kwargs: size或者scale_factor
    """
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)
class BasicConv2d1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d1, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def norm_layer(channel, norm_name='bn'):
    if norm_name == 'bn':
        return nn.BatchNorm2d(channel)
    elif norm_name == 'gn':
        return nn.GroupNorm(min(32, channel // 4), channel)


class SoftGroupingStrategy(nn.Module):
    def __init__(self, in_channel, out_channel, N):
        super(SoftGroupingStrategy, self).__init__()

        # grouping method is the only difference here
        self.g_conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[0], bias=False)
        self.g_conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[1], bias=False)
        self.g_conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[2], bias=False)

    def forward(self, q):
        # x1 = self.g_conv1(q)
        # x2 = self.g_conv2(q)
        # x3 = self.g_conv3(q)
        return self.g_conv1(q) + self.g_conv2(q) + self.g_conv3(q)




class MTG(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(MTG, self).__init__()

        self.cat3 = BasicConv2d(hidden_channels * 2, out_channels, kernel_size=3, padding=1)
        self.param_free_norm = nn.BatchNorm2d(hidden_channels, affine=False)

        self.efm1 = EFM(64)

        self.conv_res = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            norm_layer(64),
        )
        self.msca=MSCA(64)
        self.conv2d = ConvBNR(64, 64, 3)
        self.conv2d2 = ConvBNR(64, 64, 3)


    def forward(self, x, y, edge,edge_att):
        # xy = self.cat2(torch.cat((x, y), dim=1)) + y + x
        normalized = x
        # normalized=x
        edge = F.interpolate(edge, size=x.size()[2:], mode='nearest')


        out=edge+normalized
        out1=self.conv2d(out)
        wei=self.msca(out1)
        out=out1*wei
        out=self.conv2d(out+out1)

        oute=self.efm1(normalized,edge_att)

        out = self.cat3(torch.cat((out,oute), dim=1))


        return out

class CPD(nn.Module):
    def __init__(self):
        super(CPD, self).__init__()
        self.projection1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=1))

        self.projection2 = nn.Sequential(
            nn.Conv2d(64 ,64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=1))

        self.projection3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=1))

        self.predict = nn.Conv2d(64, 1, 3, 1, 1)
        self.conv4 = BasicConv2d(64,64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 1, 1)
        self.ff1=FeaFusion(64)
        self.ff2 = FeaFusion(64)
        self.ff3 = FeaFusion(64)
        self.ff4 = FeaFusion(64)


    def forward(self, x1,x2, x3, x4):

        xx3=self.ff1(x3,x4)
        pr1 = self.projection3(xx3)
        xx2 = self.ff2(x2, xx3)
        pr2 = self.projection2(xx2)
        xx1=self.ff3(x1,xx2)
        pr3 = self.projection1(xx1)

        pout1 =self.conv4(xx1)
        pout = self.conv5(pout1)

        return pout1,pout,pr1,pr2,pr3

class CropLayer(nn.Module):
    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]

class asyConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(asyConv, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
            self.initialize()
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)


    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            return square_outputs + vertical_outputs + horizontal_outputs

class RFB(nn.Module):
    def __init__(self, x, y):
        super(RFB, self).__init__()
        self.asyConv = asyConv(in_channels=x, out_channels=y, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False)
        self.oriConv = nn.Conv2d(x, y, kernel_size=3, stride=1, padding=1)
        self.atrConv = nn.Sequential(
            nn.Conv2d(x, y, kernel_size=3, dilation=3, padding=3, stride=1), nn.BatchNorm2d(y), nn.PReLU()
        )
        self.conv2d = nn.Conv2d(y*2, y, kernel_size=3, stride=1, padding=1)
        self.bn2d = nn.BatchNorm2d(y)
        self.res = BasicConv2d(x, y, 1)

    def forward(self, f):
        p2 = self.asyConv(f)
        p3 = self.atrConv(f)
        p  = torch.cat((p2, p3), 1)
        p  = F.relu(self.bn2d(self.conv2d(p)), inplace=True)

        return p

class FeaFusion(nn.Module):
    def __init__(self, channels):
        self.init__ = super(FeaFusion, self).__init__()

        self.relu = nn.ReLU()
        self.layer1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.layer2_1 = nn.Conv2d(channels, channels // 4, kernel_size=3, stride=1, padding=1)
        self.layer2_2 = nn.Conv2d(channels, channels // 4, kernel_size=3, stride=1, padding=1)

        self.layer_fu = nn.Conv2d(channels // 4, channels, kernel_size=3, stride=1, padding=1)
        self.upsample = cus_sample
        self.msca=MSCA(64)

    def forward(self, x1, x2):
        x2 = F.interpolate(x2, size=x1.shape[2:], mode='nearest')
        # wweight = nn.Sigmoid()(self.layer1(x1 + x2))
        wweight = self.msca(self.layer1(x1 + x2))
        xw_resid_1 = x1 + x1.mul(wweight)
        xw_resid_2 = x2 + x2.mul(1-wweight)
        x1_2 = self.layer2_1(xw_resid_1)
        x2_2 = self.layer2_2(xw_resid_2)
        out = self.relu(self.layer_fu(x1_2 + x2_2))

        return out
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # if self.training:
        # self.initialize_weights()
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        # self.backbone_n = pvt_v2_b2()
        path = 'pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.ed = ED()
        self.reduce1 = Conv1x1(64, 64)
        self.reduce2 = Conv1x1(128, 64)
        self.reduce3 = Conv1x1(320, 64)
        self.reduce4 = Conv1x1(512, 64)

        self.predictor1 = nn.Conv2d(64, 1, 1)
        self.predictor2 = nn.Conv2d(64, 1, 1)
        self.predictor3 = nn.Conv2d(256, 1, 1)
        # self.PSF = PSF(in_planes=512)
        self.RFB2_0 = RFB(64, 64)
        self.RFB2_1 = RFB(128, 64)
        self.RFB3_1 = RFB(320, 64)
        self.RFB4_1 = RFB(512, 64)
        self.mtg1 = MTG(64, 64)
        self.mtg2 = MTG(64, 64)
        self.mtg3 = MTG(64, 64)
        self.mtg4 = MTG(64, 64)

        self.decoder = nn.Sequential(nn.Conv2d(512* 2, 512, kernel_size=3,
                                               padding=1, stride=1, bias=False),
                                     nn.BatchNorm2d(512),
                                     nn.Conv2d(512, 1, kernel_size=1,
                                               stride=1, bias=True))
        self.cpd = CPD()
        channel=64

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)
        M = [8, 8, 8]
        N = [4, 8, 16]

        self.REM11 = REM11(64, 64)

    def forward(self, x):

        image_shape = x.size()[2:]
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        edge = self.ed(x4,x3,x2,x1)
        edge_att = torch.sigmoid(edge)
        x1_t=self.Translayer2_0(x1)
        x2_t = self.Translayer2_1(x2)  # [1, 64, 44, 44]
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)
        out,pout,pr1,pr2,pr3 = self.cpd(x1_t,x2_t, x3_t,x4_t )
        pout = torch.sigmoid(pout)

        x1r = self.RFB2_0(x1)
        x2r = self.RFB2_1(x2)
        x3r = self.RFB3_1(x3)
        x4r = self.RFB4_1(x4)
      #
        E22 = F.interpolate(x2r, scale_factor=2, mode='bilinear')
        E23 = F.interpolate(x3r, scale_factor=4, mode='bilinear')
        E24 = F.interpolate(x4r, scale_factor=8, mode='bilinear')
        # #

        R_4 = self.mtg4(E24, E24, out,edge_att)

        R_3 = self.mtg3(E23, R_4, out,edge_att)

        R_2 = self.mtg2(E22, R_3, out,edge_att)

        R_1 = self.mtg1(x1r, R_2, out,edge_att)

        pout2 = self.predictor2(R_4)
        f4, f3, f2, f1, e4, e3, e2, e1 = self.REM11([R_1, R_2, R_3, R_4], pout2, x)
        clm = F.interpolate(pout, size=image_shape, mode='bilinear')
        o1 = F.interpolate(f1, size=image_shape, mode='bilinear')
        o2 = F.interpolate(f2, size=image_shape, mode='bilinear')
        o3 = F.interpolate(f3, size=image_shape, mode='bilinear')
        o4 = F.interpolate(f4, size=image_shape, mode='bilinear')

        e1 = F.interpolate(e1, size=image_shape, mode='bilinear')
        e2 = F.interpolate(e2, size=image_shape, mode='bilinear')
        e3 = F.interpolate(e3, size=image_shape, mode='bilinear')
        e4 = F.interpolate(e4, size=image_shape, mode='bilinear')
        oe = F.interpolate(edge_att, size=image_shape, mode='bilinear')

        return clm,o1, o2, o3, o4, oe,e1,e2,e3,e4,pr1,pr2,pr3
