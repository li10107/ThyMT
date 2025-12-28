# 代码说明：这版代码和之前的区别在于HABlock聚合的是CLM之后的特征，而之前的只是聚合ResBlock输出的特征

import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import os
import torchvision.models as models
import numpy as np
# from einops import repeat
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def load_pretrained(model, premodel):
    pretrained_dict = premodel.state_dict()
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, padding=0):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class de_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(de_conv, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // scale, in_ch // scale, kernel_size=3, stride=1, padding=1)

        self.conv = de_conv(in_ch, out_ch)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x1 = self.up(x)
        x2 = self.conv(x1)
        x2 = self.dropout(x2)
        return x2


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, padding=0,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride) # H = floor((h+1)/2)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, padding=padding)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out =out + identity
        out = self.relu(out)

        return out


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, dropout=0.2):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.linear = nn.Linear(in_channels, classes, bias=True)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        y = self.activation(x)
        return y

# Baseline Plain ResUNet to segment mask and boundary
class ResUNet2dUSMaskBoundary(nn.Module):
    def __init__(self, block=BasicBlock, layers=[3, 4, 6, 3], inchannel=3, mode='multi', scale_factor=1):
        super(ResUNet2dUSMaskBoundary, self).__init__()
        self.mode=mode
        self.scale_factor = scale_factor
        self.inplanes = round(64 / self.scale_factor)
        self.conv1 = nn.Conv2d(inchannel, round(64 / self.scale_factor), kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(round(64 / self.scale_factor))
        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, round(64 / self.scale_factor), layers[0])
        self.layer2 = self._make_layer(block, round(128 / self.scale_factor), layers[1], stride=2)
        self.layer3 = self._make_layer(block, round(256 / self.scale_factor), layers[2], stride=2)
        self.layer4 = self._make_layer(block, round(512 / self.scale_factor), layers[3], stride=2)
        self.pool5 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

        self.up4_mask = up(round(512 / self.scale_factor), round(256 / self.scale_factor))
        self.up3_mask = up(round(256 * 2 / self.scale_factor), round(128 / self.scale_factor))
        self.up2_mask = up(round(128 * 2 / self.scale_factor), round(64 / self.scale_factor))
        self.up1_mask = up(round(64 * 2 / self.scale_factor), round(32 / self.scale_factor))
        self.up0_mask = up(round(32 * 2 / self.scale_factor), round(16 / self.scale_factor))
        self.outconv_mask = conv3x3(round(16 / self.scale_factor), 1)
        self.up_mask1 = up(round(64 / self.scale_factor), round(32 / self.scale_factor))

        self.up4_boundary = up(round(512 / self.scale_factor), round(256 / self.scale_factor))
        self.up3_boundary = up(round(256 * 2 / self.scale_factor), round(128 / self.scale_factor))
        self.up2_boundary = up(round(128 * 2 / self.scale_factor), round(64 / self.scale_factor))
        self.up1_boundary = up(round(64 * 2 / self.scale_factor), round(32 / self.scale_factor))
        self.up0_boundary = up(round(32 * 2 / self.scale_factor), round(16 / self.scale_factor))
        self.outconv_boundary = conv3x3(round(16 / self.scale_factor), 1)
        self.up_boundary = up(round(64 / self.scale_factor), round(32 / self.scale_factor))

        self.fcm = ClassificationHead(round(512 / self.scale_factor), 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)  # [B, 512, 9, 9]
        x5_cla = self.pool5(x5)
        y_c = self.fcm(x5_cla)

        y4_mask = self.up4_mask(x5)  # [B, 256, 18, 18]
        y4_mask = torch.cat([x4, y4_mask], dim=1)  # [B, 512, 18, 18]
        y3_mask = self.up3_mask(y4_mask)  # [B, 128, 36, 36]
        y3_mask = torch.cat([x3, y3_mask], dim=1)  # [B, 256, 36, 36]
        y2_mask = self.up2_mask(y3_mask)  # [B, 256, 72, 72]
        y2_mask = torch.cat([x2, y2_mask], dim=1)
        y1_mask = self.up1_mask(y2_mask)
        mask_1 = self.up_mask1(x1)
        y1_mask = torch.cat([mask_1, y1_mask], dim=1)
        y0_mask = self.up0_mask(y1_mask)
        y_mask = self.outconv_mask(y0_mask)

        y4_boundary= self.up4_boundary(x5)
        y4_boundary = torch.cat([x4, y4_boundary], dim=1)
        y3_boundary = self.up3_boundary(y4_boundary)
        y3_boundary = torch.cat([x3, y3_boundary], dim=1)
        y2_boundary = self.up2_boundary(y3_boundary)
        y2_boundary = torch.cat([x2, y2_boundary], dim=1)
        y1_boundary = self.up1_boundary(y2_boundary)
        boundary1 = self.up_boundary(x1)
        y1_boundary = torch.cat([boundary1, y1_boundary], dim=1)
        y0_boundary = self.up0_boundary(y1_boundary)
        y_boundary = self.outconv_boundary(y0_boundary)
        if self.mode == 'multi':
            return y_mask, y_boundary, x5_cla
        elif self.mode == 'uni':
            return y_mask, y_boundary, y_c

# Ablation 2_1: Plain ResUNet + hierarchical Attention Fusion

class HierachicalAttention(nn.Module):
    def __init__(self):
        super(HierachicalAttention, self).__init__()
        self.dim = 512
        self.key3 = conv1x1(128, self.dim//2)
        self.value3 = nn.Sequential(conv1x1(128, self.dim), nn.BatchNorm2d(self.dim), nn.ReLU())
        self.key4 = conv1x1(256, self.dim//2)
        self.value4 = nn.Sequential(conv1x1(256, self.dim), nn.BatchNorm2d(self.dim), nn.ReLU())
        self.key5 = conv1x1(512, self.dim//2)
        self.value5 = nn.Sequential(conv1x1(512, self.dim), nn.BatchNorm2d(self.dim), nn.ReLU())
        self.query5 = nn.Sequential(conv1x1(512, self.dim//2), nn.AdaptiveAvgPool2d(1))

        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Sequential(nn.Linear(self.dim, 1, bias=True), nn.Sigmoid())


    def forward(self, x3, x4, x5):
        B3, C3, H3, W3 = x3.size()
        key3 = self.key3(x3).view(B3, -1, H3 * W3)  # [B, 256, 72, 72]-> [B, 256, 72*72]
        value3 = self.value3(x3).view(B3, -1, H3 * W3)    # [B, 512, 72, 72]-> [B, 512, 72*72]

        B4, C4, H4, W4 = x4.size()
        key4 = self.key4(x4).view(B4, -1, H4 * W4)  # [B, 256, 18, 18] -> [B, 256, 18*18]
        value4 = self.value4(x4).view(B4, -1, H4 * W4)   # [B, 512, 18, 18]

        B5, C5, H5, W5 = x5.size()
        key5 = self.key5(x5).view(B5, -1, H5 * W5)  # [B, 256, 9, 9]-> [B, 256, 9*9]
        value5 = self.value5(x5).view(B5, -1, H5 * W5)   # [B, 512, 9*9]

        query5 = self.query5(x5).view(B5, -1, 1*1)  # [B, 256, 1*1]

        key = torch.cat([key3, key4, key5], dim=2)  # [B, 256, 72*72+18*18+9*9]
        value = torch.cat([value3, value4, value5], dim=2)  # [B, 512, 72*72+18*18+9*9]
        query = query5.permute(0, 2, 1)  # [B, 256, 1*1] -> [B, 1, 256]

        energy = torch.bmm(query, key)  # [B, 1, 256] * [B, 256, N] -> [B, 1, N]
        energy = energy / math.sqrt(self.dim)
        attention = self.softmax(energy).permute(0, 2, 1)  # [B, 1, N]，在N维度进行归一化 -> [B, N, 1]

        out = torch.bmm(value, attention).squeeze(2)  # [B, 512, N] * [B, N, 1] -> [B, 512, 1]
        y_c = self.fc(out)
        return y_c


class ResUNet2dUSMaskBoundaryHierachicalAttention(nn.Module):
    def __init__(self, block=BasicBlock, layers=[3, 4, 6, 3], inchannel=3, mode='multi', scale_factor=1):
        super(ResUNet2dUSMaskBoundaryHierachicalAttention, self).__init__()
        self.mode=mode
        self.scale_factor = scale_factor
        self.inplanes = round(64 / self.scale_factor)
        self.conv1 = nn.Conv2d(inchannel, round(64 / self.scale_factor), kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(round(64 / self.scale_factor))
        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, round(64 / self.scale_factor), layers[0])
        self.layer2 = self._make_layer(block, round(128 / self.scale_factor), layers[1], stride=2)
        self.layer3 = self._make_layer(block, round(256 / self.scale_factor), layers[2], stride=2)
        self.layer4 = self._make_layer(block, round(512 / self.scale_factor), layers[3], stride=2)
        self.HierachicalBlock = HierachicalAttention()

        self.up4_mask = up(round(512 / self.scale_factor), round(256 / self.scale_factor))
        self.up3_mask = up(round(256 * 2 / self.scale_factor), round(128 / self.scale_factor))
        self.up2_mask = up(round(128 * 2 / self.scale_factor), round(64 / self.scale_factor))
        self.up1_mask = up(round(64 * 2 / self.scale_factor), round(32 / self.scale_factor))
        self.up0_mask = up(round(32 * 2 / self.scale_factor), round(16 / self.scale_factor))
        self.outconv_mask = conv3x3(round(16 / self.scale_factor), 1)
        self.up_mask1 = up(round(64 / self.scale_factor), round(32 / self.scale_factor))

        self.up4_boundary = up(round(512 / self.scale_factor), round(256 / self.scale_factor))
        self.up3_boundary = up(round(256 * 2 / self.scale_factor), round(128 / self.scale_factor))
        self.up2_boundary = up(round(128 * 2 / self.scale_factor), round(64 / self.scale_factor))
        self.up1_boundary = up(round(64 * 2 / self.scale_factor), round(32 / self.scale_factor))
        self.up0_boundary = up(round(32 * 2 / self.scale_factor), round(16 / self.scale_factor))
        self.outconv_boundary = conv3x3(round(16 / self.scale_factor), 1)
        self.up_boundary = up(round(64 / self.scale_factor), round(32 / self.scale_factor))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)  # [B, 64, 72, 72]
        x3 = self.layer2(x2)  # [B, 128, 36, 36]
        x4 = self.layer3(x3)  # [B, 256, 18, 18]
        x5 = self.layer4(x4)  # [B, 512, 9, 9]

        y4_mask = self.up4_mask(x5)  # [B, 256, 18, 18]
        y4_mask = torch.cat([x4, y4_mask], dim=1)  # [B, 512, 18, 18]
        y3_mask = self.up3_mask(y4_mask)  # [B, 128, 36, 36]
        y3_mask = torch.cat([x3, y3_mask], dim=1)  # [B, 256, 36, 36]
        y2_mask = self.up2_mask(y3_mask)  # [B, 256, 72, 72]
        y2_mask = torch.cat([x2, y2_mask], dim=1)
        y1_mask = self.up1_mask(y2_mask)
        mask_1 = self.up_mask1(x1)
        y1_mask = torch.cat([mask_1, y1_mask], dim=1)
        y0_mask = self.up0_mask(y1_mask)
        y_mask = self.outconv_mask(y0_mask)

        y4_boundary= self.up4_boundary(x5)
        y4_boundary = torch.cat([x4, y4_boundary], dim=1)
        y3_boundary = self.up3_boundary(y4_boundary)
        y3_boundary = torch.cat([x3, y3_boundary], dim=1)
        y2_boundary = self.up2_boundary(y3_boundary)
        y2_boundary = torch.cat([x2, y2_boundary], dim=1)
        y1_boundary = self.up1_boundary(y2_boundary)
        boundary1 = self.up_boundary(x1)
        y1_boundary = torch.cat([boundary1, y1_boundary], dim=1)
        y0_boundary = self.up0_boundary(y1_boundary)
        y_boundary = self.outconv_boundary(y0_boundary)

        y_c = self.HierachicalBlock(x3, x4, x5)
        return y_mask, y_boundary, y_c

def resunet_us_34_mask_texture(mode='uni', pretrain_flag=True):
    resunet_us = ResUNet2dUSMaskBoundary(mode=mode)
    if pretrain_flag == True:
        premodel = models.resnet34(pretrained=pretrain_flag)
        resunet_us = load_pretrained(resunet_us, premodel)
    return resunet_us


def HierachicalAttentionNet(pretrain_flag=True, backbone='ResNet34'):
    if backbone == 'ResNet34':
        net = ResUNet2dUSMaskBoundaryHierachicalAttention()
    elif backbone == 'ResNet18':
        net = ResUNet2dUSMaskBoundaryHierachicalAttention(layers=[2, 2, 2, 2])
    if pretrain_flag == True:
        if backbone == 'ResNet34':
            premodel = models.resnet34(pretrained=pretrain_flag)
        elif backbone == 'ResNet18':
            premodel = models.resnet18(pretrained=pretrain_flag)
        net = load_pretrained(net, premodel)
    return net


# 0718 idea: 利用底层分割特征指导高层分割

class CrossLayerAttention(nn.Module):
    def __init__(self, dim0, dim1):
        super(CrossLayerAttention, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        self.dim_query = 256
        self.query0 = nn.Sequential(conv1x1(self.dim0, self.dim_query), nn.AdaptiveAvgPool2d(1))  # 第0层的feature map的通道数是第1层的2倍
        self.key1 = conv1x1(self.dim1, self.dim_query)
        self.value1 = nn.Sequential(conv1x1(self.dim1, self.dim1), nn.BatchNorm2d(self.dim1), nn.ReLU())
        self.softmax = nn.Softmax(dim=-1)
        self.conv_post = nn.Sequential(conv1x1(self.dim1, self.dim1), nn.BatchNorm2d(self.dim1), nn.ReLU())
        self.alpha = nn.Parameter(torch.randn(1))

    def forward(self, x0, x1):
        B0, C0, H0, W0 = x0.size()
        query0 = self.query0(x0).view(B0, -1, 1*1).permute(0, 2, 1)  # [B0, dim_query, 1] -> [B0, 1, dim_query]

        B1, C1, H1, W1 = x1.size()
        key1 = self.key1(x1).view(B1, -1, H1 * W1)  # [B1, dim_query, H1 * W1]
        value1 = self.value1(x1)  # [B1, dim1, H1, W1]

        energy = torch.bmm(query0, key1)   # [B0, 1, dim_query] * [B0, dim_query, H1 * W1] -> [B0, 1, H1 * W1]
        energy = energy / math.sqrt(self.dim_query)  # [B0, 1, H1 * W1]
        attention = self.softmax(energy).view(B1, 1, H1, W1)  # [B0, 1, H1, W1]

        seg_enhance = value1 * attention
        out = x1 + seg_enhance
        out_post = self.conv_post(out)
        seg_enhance = self.alpha * seg_enhance
        return out_post, seg_enhance


class ResUNet2dUSMaskBoundaryCrossLayer(nn.Module):
    def __init__(self, block=BasicBlock, layers=[3, 4, 6, 3], inchannel=3, mode='multi', scale_factor=1):
        super(ResUNet2dUSMaskBoundaryCrossLayer, self).__init__()
        self.mode=mode
        self.scale_factor = scale_factor
        self.inplanes = round(64 / self.scale_factor)
        self.conv1 = nn.Conv2d(inchannel, round(64 / self.scale_factor), kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(round(64 / self.scale_factor))
        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, round(64 / self.scale_factor), layers[0])
        self.layer2 = self._make_layer(block, round(128 / self.scale_factor), layers[1], stride=2)
        self.layer3 = self._make_layer(block, round(256 / self.scale_factor), layers[2], stride=2)
        self.layer4 = self._make_layer(block, round(512 / self.scale_factor), layers[3], stride=2)
        self.pool5 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

        self.map1_mask = nn.Sequential(conv1x1(64, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.map1_boundary = nn.Sequential(conv1x1(64, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.CrossLayer12_mask = CrossLayerAttention(64, 128)
        self.CrossLayer23_mask = CrossLayerAttention(128, 256)
        # self.CrossLayer34_mask = CrossLayerAttention(256, 512)
        self.CrossLayer12_boundary = CrossLayerAttention(64, 128)
        self.CrossLayer23_boundary = CrossLayerAttention(128, 256)
        # self.CrossLayer34_boundary = CrossLayerAttention(256, 512)

        self.up4_mask = up(round(512 / self.scale_factor), round(256 / self.scale_factor))
        self.up3_mask = up(round(256 * 2 / self.scale_factor), round(128 / self.scale_factor))
        self.up2_mask = up(round(128 * 2 / self.scale_factor), round(64 / self.scale_factor))
        self.up1_mask = up(round(64 * 2 / self.scale_factor), round(32 / self.scale_factor))
        self.up0_mask = up(round(32 * 2 / self.scale_factor), round(16 / self.scale_factor))
        self.outconv_mask = conv3x3(round(16 / self.scale_factor), 1)
        self.up_mask1 = up(round(64 / self.scale_factor), round(32 / self.scale_factor))

        self.up4_boundary = up(round(512 / self.scale_factor), round(256 / self.scale_factor))
        self.up3_boundary = up(round(256 * 2 / self.scale_factor), round(128 / self.scale_factor))
        self.up2_boundary = up(round(128 * 2 / self.scale_factor), round(64 / self.scale_factor))
        self.up1_boundary = up(round(64 * 2 / self.scale_factor), round(32 / self.scale_factor))
        self.up0_boundary = up(round(32 * 2 / self.scale_factor), round(16 / self.scale_factor))
        self.outconv_boundary = conv3x3(round(16 / self.scale_factor), 1)
        self.up_boundary = up(round(64 / self.scale_factor), round(32 / self.scale_factor))
        self.fcm = ClassificationHead(round(512 / self.scale_factor), 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)  # [B, 64, 72, 72]
        x2_mask = self.map1_mask(x2)  # [B, 64, 72, 72]
        x2_boundary = self.map1_boundary(x2)  # [B, 64, 72, 72]
        x3 = self.layer2(x2)  # [B, 128, 36, 36]
        out_post_mask_3, mask_hence3 = self.CrossLayer12_mask(x2_mask, x3) # x2_mask取全局池化，指导x3中mask特征增强，将增强后的特征和mask特征相加，送入后续分类，将增强mask特征和原始特征相加，继续分类
        out_post_boundary3, boundary_hence3 = self.CrossLayer12_boundary(x2_boundary, x3)
        x4 = self.layer3(x3 + mask_hence3 + boundary_hence3)  # [B, 256, 18, 18]
        out_post_mask_4, mask_hence4 = self.CrossLayer23_mask(out_post_mask_3, x4)
        out_post_boundary4, boundary_hence4 = self.CrossLayer23_boundary(out_post_boundary3, x4)
        x5 = self.layer4(x4 + mask_hence4 + boundary_hence4)  # [B, 512, 9, 9]

        x5_cla = self.pool5(x5)  # [B, 512]
        y_c = self.fcm(x5_cla)

        y4_mask = self.up4_mask(x5)  # [B, 256, 18, 18]
        y4_mask = torch.cat([out_post_mask_4, y4_mask], dim=1)  # [B, 512, 18, 18]
        y3_mask = self.up3_mask(y4_mask)  # [B, 128, 36, 36]
        y3_mask = torch.cat([out_post_mask_3, y3_mask], dim=1)  # [B, 256, 36, 36]
        y2_mask = self.up2_mask(y3_mask)  # [B, 256, 72, 72]
        y2_mask = torch.cat([x2_mask, y2_mask], dim=1)
        y1_mask = self.up1_mask(y2_mask)
        mask_1 = self.up_mask1(x1)
        y1_mask = torch.cat([mask_1, y1_mask], dim=1)
        y0_mask = self.up0_mask(y1_mask)
        y_mask = self.outconv_mask(y0_mask)

        y4_boundary= self.up4_boundary(x5)
        y4_boundary = torch.cat([out_post_boundary4, y4_boundary], dim=1)
        y3_boundary = self.up3_boundary(y4_boundary)
        y3_boundary = torch.cat([out_post_boundary3, y3_boundary], dim=1)
        y2_boundary = self.up2_boundary(y3_boundary)
        y2_boundary = torch.cat([x2_boundary, y2_boundary], dim=1)
        y1_boundary = self.up1_boundary(y2_boundary)
        boundary1 = self.up_boundary(x1)
        y1_boundary = torch.cat([boundary1, y1_boundary], dim=1)
        y0_boundary = self.up0_boundary(y1_boundary)
        y_boundary = self.outconv_boundary(y0_boundary)

        return y_mask, y_boundary, y_c


def ResUNet2dUSMaskBoundaryCrossLayerNet(pretrain_flag=True, backbone='ResNet34'):
    if backbone == 'ResNet34':
        net = ResUNet2dUSMaskBoundaryCrossLayer()
    elif backbone == 'ResNet18':
        net = ResUNet2dUSMaskBoundaryCrossLayer(layers=[2, 2, 2, 2])
    if pretrain_flag == True:
        if backbone == 'ResNet34':
            premodel = models.resnet34(pretrained=pretrain_flag)
        elif backbone == 'ResNet18':
            premodel = models.resnet18(pretrained=pretrain_flag)
        net = load_pretrained(net, premodel)
    return net


class ResUNet2dUSMaskBoundaryCrossLayerHierachicalAttention(nn.Module):
    def __init__(self, block=BasicBlock, layers=[3, 4, 6, 3], inchannel=3, mode='multi', scale_factor=1):
        super(ResUNet2dUSMaskBoundaryCrossLayerHierachicalAttention, self).__init__()
        self.mode=mode
        self.scale_factor = scale_factor
        self.inplanes = round(64 / self.scale_factor)
        self.conv1 = nn.Conv2d(inchannel, round(64 / self.scale_factor), kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(round(64 / self.scale_factor))
        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, round(64 / self.scale_factor), layers[0])
        self.layer2 = self._make_layer(block, round(128 / self.scale_factor), layers[1], stride=2)
        self.layer3 = self._make_layer(block, round(256 / self.scale_factor), layers[2], stride=2)
        self.layer4_bm = self._make_layer(block, round(512 / self.scale_factor), layers[3], stride=2, expansion=False)
        self.layer4_lateral = self._make_layer(block, round(512 / self.scale_factor), layers[3], stride=2, expansion=False)
        self.layer4_center = self._make_layer(block, round(512 / self.scale_factor), layers[3], stride=2, expansion=False)

        self.map1_mask = nn.Sequential(conv1x1(64, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.map1_boundary = nn.Sequential(conv1x1(64, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.CrossLayer12_mask = CrossLayerAttention(64, 128)
        self.CrossLayer23_mask = CrossLayerAttention(128, 256)
        # self.CrossLayer34_mask = CrossLayerAttention(256, 512)
        self.CrossLayer12_boundary = CrossLayerAttention(64, 128)
        self.CrossLayer23_boundary = CrossLayerAttention(128, 256)
        # self.CrossLayer34_boundary = CrossLayerAttention(256, 512)
        self.HierachicalBlock_bm = HierachicalAttention()
        self.HierachicalBlock_lateral = HierachicalAttention()
        self.HierachicalBlock_center = HierachicalAttention()

        self.up4_mask = up(round(512 / self.scale_factor), round(256 / self.scale_factor))
        self.up3_mask = up(round(256 * 2 / self.scale_factor), round(128 / self.scale_factor))
        self.up2_mask = up(round(128 * 2 / self.scale_factor), round(64 / self.scale_factor))
        self.up1_mask = up(round(64 * 2 / self.scale_factor), round(32 / self.scale_factor))
        self.up0_mask = up(round(32 * 2 / self.scale_factor), round(16 / self.scale_factor))
        self.outconv_mask = conv3x3(round(16 / self.scale_factor), 1)
        self.up_mask1 = up(round(64 / self.scale_factor), round(32 / self.scale_factor))

        self.up4_boundary = up(round(512 / self.scale_factor), round(256 / self.scale_factor))
        self.up3_boundary = up(round(256 * 2 / self.scale_factor), round(128 / self.scale_factor))
        self.up2_boundary = up(round(128 * 2 / self.scale_factor), round(64 / self.scale_factor))
        self.up1_boundary = up(round(64 * 2 / self.scale_factor), round(32 / self.scale_factor))
        self.up0_boundary = up(round(32 * 2 / self.scale_factor), round(16 / self.scale_factor))
        self.outconv_boundary = conv3x3(round(16 / self.scale_factor), 1)
        self.up_boundary = up(round(64 / self.scale_factor), round(32 / self.scale_factor))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, expansion=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        if not expansion:
            self.inplanes = 256

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)  # [B, 64, 72, 72]
        x2_mask = self.map1_mask(x2)  # [B, 64, 72, 72]
        x2_boundary = self.map1_boundary(x2)  # [B, 64, 72, 72]
        x3 = self.layer2(x2)  # [B, 128, 36, 36]
        out_post_mask_3, mask_hence3 = self.CrossLayer12_mask(x2_mask, x3) # x2_mask取全局池化，指导x3中mask特征增强，将增强后的特征和mask特征相加，送入后续分类，将增强mask特征和原始特征相加，继续分类
        out_post_boundary3, boundary_hence3 = self.CrossLayer12_boundary(x2_boundary, x3)
        x4 = self.layer3(x3 + mask_hence3 + boundary_hence3)  # [B, 256, 18, 18]
        out_post_mask_4, mask_hence4 = self.CrossLayer23_mask(out_post_mask_3, x4)
        out_post_boundary4, boundary_hence4 = self.CrossLayer23_boundary(out_post_boundary3, x4)
        x5_bm = self.layer4_bm(x4 + mask_hence4 + boundary_hence4)  # [B, 512, 9, 9]
        x5_lateral = self.layer4_lateral(x4 + mask_hence4 + boundary_hence4)  # [B, 512, 9, 9]
        x5_center = self.layer4_center(x4 + mask_hence4 + boundary_hence4)  # [B, 512, 9, 9]

        x5_decode = x5_bm + x5_lateral + x5_center

        y4_mask = self.up4_mask(x5_decode)  # [B, 256, 18, 18]
        y4_mask = torch.cat([out_post_mask_4, y4_mask], dim=1)  # [B, 512, 18, 18]
        y3_mask = self.up3_mask(y4_mask)  # [B, 128, 36, 36]
        y3_mask = torch.cat([out_post_mask_3, y3_mask], dim=1)  # [B, 256, 36, 36]
        y2_mask = self.up2_mask(y3_mask)  # [B, 256, 72, 72]
        y2_mask = torch.cat([x2_mask, y2_mask], dim=1)
        y1_mask = self.up1_mask(y2_mask)
        mask_1 = self.up_mask1(x1)
        y1_mask = torch.cat([mask_1, y1_mask], dim=1)
        y0_mask = self.up0_mask(y1_mask)
        y_mask = self.outconv_mask(y0_mask)

        y4_boundary= self.up4_boundary(x5_decode)
        y4_boundary = torch.cat([out_post_boundary4, y4_boundary], dim=1)
        y3_boundary = self.up3_boundary(y4_boundary)
        y3_boundary = torch.cat([out_post_boundary3, y3_boundary], dim=1)
        y2_boundary = self.up2_boundary(y3_boundary)
        y2_boundary = torch.cat([x2_boundary, y2_boundary], dim=1)
        y1_boundary = self.up1_boundary(y2_boundary)
        boundary1 = self.up_boundary(x1)
        y1_boundary = torch.cat([boundary1, y1_boundary], dim=1)
        y0_boundary = self.up0_boundary(y1_boundary)
        y_boundary = self.outconv_boundary(y0_boundary)

        y_c_bm = self.HierachicalBlock_bm(x3 + mask_hence3 + boundary_hence3, x4 + mask_hence4 + boundary_hence4, x5_bm)
        y_c_lateral = self.HierachicalBlock_lateral(x3 + mask_hence3 + boundary_hence3, x4 + mask_hence4 + boundary_hence4, x5_lateral)
        y_c_center = self.HierachicalBlock_center(x3 + mask_hence3 + boundary_hence3, x4 + mask_hence4 + boundary_hence4, x5_center)
        return y_mask, y_boundary, y_c_bm, y_c_lateral, y_c_center


class ResUNet2dUSMaskCrossLayerHierachicalAttention(nn.Module):
    def __init__(self, block=BasicBlock, layers=[3, 4, 6, 3], inchannel=3, mode='multi', scale_factor=1):
        super(ResUNet2dUSMaskCrossLayerHierachicalAttention, self).__init__()
        self.mode=mode
        self.scale_factor = scale_factor
        self.inplanes = round(64 / self.scale_factor)
        self.conv1 = nn.Conv2d(inchannel, round(64 / self.scale_factor), kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(round(64 / self.scale_factor))
        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, round(64 / self.scale_factor), layers[0])
        self.layer2 = self._make_layer(block, round(128 / self.scale_factor), layers[1], stride=2)
        self.layer3 = self._make_layer(block, round(256 / self.scale_factor), layers[2], stride=2)
        self.layer4 = self._make_layer(block, round(512 / self.scale_factor), layers[3], stride=2)

        self.map1_mask = nn.Sequential(conv1x1(64, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.CrossLayer12_mask = CrossLayerAttention(64, 128)
        self.CrossLayer23_mask = CrossLayerAttention(128, 256)
        # self.CrossLayer34_mask = CrossLayerAttention(256, 512)
        self.HierachicalBlock = HierachicalAttention()

        self.up4_mask = up(round(512 / self.scale_factor), round(256 / self.scale_factor))
        self.up3_mask = up(round(256 * 2 / self.scale_factor), round(128 / self.scale_factor))
        self.up2_mask = up(round(128 * 2 / self.scale_factor), round(64 / self.scale_factor))
        self.up1_mask = up(round(64 * 2 / self.scale_factor), round(32 / self.scale_factor))
        self.up0_mask = up(round(32 * 2 / self.scale_factor), round(16 / self.scale_factor))
        self.outconv_mask = conv3x3(round(16 / self.scale_factor), 1)
        self.up_mask1 = up(round(64 / self.scale_factor), round(32 / self.scale_factor))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)  # [B, 64, 72, 72]
        x2_mask = self.map1_mask(x2)  # [B, 64, 72, 72]
        x3 = self.layer2(x2)  # [B, 128, 36, 36]
        out_post_mask_3, mask_hence3 = self.CrossLayer12_mask(x2_mask, x3) # x2_mask取全局池化，指导x3中mask特征增强，将增强后的特征和mask特征相加，送入后续分类，将增强mask特征和原始特征相加，继续分类
        x4 = self.layer3(x3 + mask_hence3)  # [B, 256, 18, 18]
        out_post_mask_4, mask_hence4 = self.CrossLayer23_mask(out_post_mask_3, x4)
        x5 = self.layer4(x4 + mask_hence4)  # [B, 512, 9, 9]

        y4_mask = self.up4_mask(x5)  # [B, 256, 18, 18]
        y4_mask = torch.cat([out_post_mask_4, y4_mask], dim=1)  # [B, 512, 18, 18]
        y3_mask = self.up3_mask(y4_mask)  # [B, 128, 36, 36]
        y3_mask = torch.cat([out_post_mask_3, y3_mask], dim=1)  # [B, 256, 36, 36]
        y2_mask = self.up2_mask(y3_mask)  # [B, 256, 72, 72]
        y2_mask = torch.cat([x2_mask, y2_mask], dim=1)
        y1_mask = self.up1_mask(y2_mask)
        mask_1 = self.up_mask1(x1)
        y1_mask = torch.cat([mask_1, y1_mask], dim=1)
        y0_mask = self.up0_mask(y1_mask)
        y_mask = self.outconv_mask(y0_mask)

        y_c = self.HierachicalBlock(x3 + mask_hence3, x4 + mask_hence4, x5)
        return y_mask, y_c


def ResUNet2dUSMaskBoundaryCrossLayerHierachicalAttentionNet(pretrain_flag=True, backbone='ResNet34'):
    if backbone == 'ResNet34':
        net = ResUNet2dUSMaskBoundaryCrossLayerHierachicalAttention()
    elif backbone == 'ResNet18':
        net = ResUNet2dUSMaskBoundaryCrossLayerHierachicalAttention(layers=[2, 2, 2, 2])
    if pretrain_flag == True:
        if backbone == 'ResNet34':
            premodel = models.resnet34(pretrained=pretrain_flag)
        elif backbone == 'ResNet18':
            premodel = models.resnet18(pretrained=pretrain_flag)
        net = load_pretrained(net, premodel)
    return net


def ResUNet2dUSMaskCrossLayerHierachicalAttentionNet(pretrain_flag=True, backbone='ResNet34'):
    if backbone == 'ResNet34':
        net = ResUNet2dUSMaskCrossLayerHierachicalAttention()
    elif backbone == 'ResNet18':
        net = ResUNet2dUSMaskCrossLayerHierachicalAttention(layers=[2, 2, 2, 2])
    if pretrain_flag == True:
        if backbone == 'ResNet34':
            premodel = models.resnet34(pretrained=pretrain_flag)
        elif backbone == 'ResNet18':
            premodel = models.resnet18(pretrained=pretrain_flag)
        net = load_pretrained(net, premodel)
    return net


if __name__ == '__main__':
    a = torch.randn(2, 3, 288, 288)
    net = ResUNet2dUSMaskBoundaryCrossLayerHierachicalAttentionNet()
    a = a.cuda()
    net.cuda()
    y_mask, y_boundary, y_c = net(a)
    print(y_mask.shape)
    print(y_boundary.shape)
    print(y_c.shape)