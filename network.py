import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
import copy

#content_layers_default = ['conv_4']
#style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True):
        super(ConvLayer, self).__init__()
        padd = int(math.floor(kernel_size / 2))
        self.reflect = nn.ReflectionPad2d(padd)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

    def forward(self, x):
        x = self.reflect(x)
        x = self.conv(x)
        return x


class ConvInstReLU(ConvLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvInstReLU, self).__init__(in_channels, out_channels, kernel_size, stride)
        self.inst = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = super(ConvInstReLU, self).forward(x)
        x = self.inst(x)
        x = self.relu(x)
        return x


class ConvTanh(ConvLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvTanh, self).__init__(in_channels, out_channels, kernel_size, stride)
        self.tanh = nn.Tanh()
#        self.tanh = nn.Sigmoid()

    def forward(self, x):
        x = super(ConvTanh, self).forward(x)
#        x = self.tanh(x/255) * 150 + 255/2
        x = self.tanh(x)
#        x = (x + 1) / 2
#        x[x<0]=0
#        x[x>1]=1
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=padding)
        self.inst1 = nn.InstanceNorm2d(out_channel, affine=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding=padding)
        self.inst2 = nn.InstanceNorm2d(out_channel, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.relu(self.inst1(self.conv1(x)))
        x = self.inst2(self.conv2(x))
        x = res + x
        return x



class ReCoNet(nn.Module):
    def __init__(self):
        super(ReCoNet, self).__init__()
        self.cir1 = ConvInstReLU(3, 32, 9, 1)
        self.cir2 = ConvInstReLU(32, 64, 3, 2)
        self.cir3 = ConvInstReLU(64, 128, 3, 2)

        self.rir1 = ResidualBlock(128, 128)
        self.rir2 = ResidualBlock(128, 128)
        self.rir3 = ResidualBlock(128, 128)
        self.rir4 = ResidualBlock(128, 128)
        self.rir5 = ResidualBlock(128, 128)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.devcir1 = ConvInstReLU(128, 64, 3, 1)
        self.devcir2 = ConvInstReLU(64, 32, 3, 1)

        self.tanh = ConvTanh(32, 3, 9, 1)
        self.deconv = ConvLayer(32, 3, 9, 1)

    def forward(self, x):
        x = self.cir1(x)
        x = self.cir2(x)
        x = self.cir3(x)

        x = self.rir1(x)
        x = self.rir2(x)
        x = self.rir3(x)
        x = self.rir4(x)
        x = self.rir5(x)

        feat = x

        x = self.up1(x)
        x = self.devcir1(x)
        x = self.up2(x)
        x = self.devcir2(x)
        x = self.tanh(x)
#        x = self.deconv(x)

        return feat, x


class Test(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Test, self).__init__()
        self.aa = 1

    def forward(self, y):
        output = y + 1
        print('Test:')
        return output


class Test2(Test):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Test2, self).__init__(in_channels, out_channels, kernel_size, stride)
        self.abc = 1
        print('Test2')
        # nn.Conv2d

    def forward(self, y):
        output = super(Test2, self).forward(y)
        return output + 1


def gram(input):
    a, b, c, d = input.size()
    gram = input.view(a * b, c * d)
    G = torch.mm(gram, gram.T)
    return G / (a * b * c * d)


def gram2(input):
    a, b, c, d = input.size()  ####a>=2
    list = []
    for i in range(a):
        sample = input[i]
        gram = sample.view(b, c * d)
        G = torch.mm(gram, gram.T)
        list.append(G.unsqueeze(0))
    return torch.cat(list, dim=0) / (b * c * d)

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True, progress=True).features.eval()
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        for i in range(4):
            self.slice1.add_module(str(i), vgg[i])
        for i in range(4, 9):
            self.slice2.add_module(str(i), vgg[i])
        for i in range(9, 16):
            self.slice3.add_module(str(i), vgg[i])
        for i in range(16, 23):
            self.slice4.add_module(str(i), vgg[i])

    def forward(self, x):

        x = self.slice1(x)
        relu1_2 = x
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_3 = x
        x = self.slice4(x)
        relu4_3 = x

        return relu1_2, relu2_2, relu3_3, relu4_3


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
#        img = img.div_(255.0)
        return (img - self.mean) / self.std


def main():
    input = torch.randn(1, 3, 640, 360)
    model = ReCoNet()
    feat, output = model(input)
    print(feat.size(), output.size())


if __name__ == '__main__':
    main()

