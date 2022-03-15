from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import sys

__all__ = ['InceptionResNetV2', 'inceptionresnetv2']

pretrained_settings = {
    'inceptionresnetv2': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
        'imagenet+background': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1001
        }
    }
}


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out1 = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out1)
        out = out * self.scale + x
        out = self.relu(out)
        return out, out1


class Mixed_6a(nn.Module):
    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(160, 192, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out1 = torch.cat((x0, x1), 1)
        out = self.conv2d(out1)
        out = out * self.scale + x
        out = self.relu(out)
        return out, out1


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(2080, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(224, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out1 = torch.cat((x0, x1), 1)
        out = self.conv2d(out1)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out, out1


class InceptionResNetV2(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionResNetV2, self).__init__()
        # Special attributs
        self.input_space = None
        # self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        # self.repeat = nn.Sequential(
        #     Block35(scale=0.17),
        #     Block35(scale=0.17),
        #     Block35(scale=0.17),
        #     Block35(scale=0.17),
        #     Block35(scale=0.17),
        #     Block35(scale=0.17),
        #     Block35(scale=0.17),
        #     Block35(scale=0.17),
        #     Block35(scale=0.17),
        #     Block35(scale=0.17)
        # )
        self.Block35_1 = Block35(scale=0.17)
        self.Block35_2 = Block35(scale=0.17)
        self.Block35_3 = Block35(scale=0.17)
        self.Block35_4 = Block35(scale=0.17)
        self.Block35_5 = Block35(scale=0.17)
        self.Block35_6 = Block35(scale=0.17)
        self.Block35_7 = Block35(scale=0.17)
        self.Block35_8 = Block35(scale=0.17)
        self.Block35_9 = Block35(scale=0.17)
        self.Block35_10 = Block35(scale=0.17)

        self.mixed_6a = Mixed_6a()

        self.Block17_1 = Block17(scale=0.10)
        self.Block17_2 = Block17(scale=0.10)
        self.Block17_3 = Block17(scale=0.10)
        self.Block17_4 = Block17(scale=0.10)
        self.Block17_5 = Block17(scale=0.10)
        self.Block17_6 = Block17(scale=0.10)
        self.Block17_7 = Block17(scale=0.10)
        self.Block17_8 = Block17(scale=0.10)
        self.Block17_9 = Block17(scale=0.10)
        self.Block17_10 = Block17(scale=0.10)
        self.Block17_11 = Block17(scale=0.10)
        self.Block17_12 = Block17(scale=0.10)
        self.Block17_13 = Block17(scale=0.10)
        self.Block17_14 = Block17(scale=0.10)
        self.Block17_15 = Block17(scale=0.10)
        self.Block17_16 = Block17(scale=0.10)
        self.Block17_17 = Block17(scale=0.10)
        self.Block17_18 = Block17(scale=0.10)
        self.Block17_19 = Block17(scale=0.10)
        self.Block17_20 = Block17(scale=0.10)

        self.mixed_7a = Mixed_7a()

        self.Block8_1 = Block8(scale=0.20)
        self.Block8_2 = Block8(scale=0.20)
        self.Block8_3 = Block8(scale=0.20)
        self.Block8_4 = Block8(scale=0.20)
        self.Block8_5 = Block8(scale=0.20)
        self.Block8_6 = Block8(scale=0.20)
        self.Block8_7 = Block8(scale=0.20)
        self.Block8_8 = Block8(scale=0.20)
        self.Block8_9 = Block8(scale=0.20)

        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(5)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        # self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)

    def features(self, input):
        x = self.conv2d_1a(input)
        # print(x.shape)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        # mixed 5b
        x = self.mixed_5b(x)
        a = self.avgpool(x)
        # block 35
        x, feature1 = self.Block35_1(x)
        b_1 = self.avgpool(feature1)
        x, feature2 = self.Block35_2(x)
        b_2 = self.avgpool(feature2)
        x, feature3 = self.Block35_3(x)
        b_3 = self.avgpool(feature3)
        x, feature4 = self.Block35_4(x)
        b_4 = self.avgpool(feature4)
        x, feature5 = self.Block35_5(x)
        b_5 = self.avgpool(feature5)
        x, feature6 = self.Block35_6(x)
        b_6 = self.avgpool(feature6)
        x, feature7 = self.Block35_7(x)
        b_7 = self.avgpool(feature7)
        x, feature8 = self.Block35_8(x)
        b_8 = self.avgpool(feature8)
        x, feature9 = self.Block35_9(x)
        b_9 = self.avgpool(feature9)
        x, feature10 = self.Block35_10(x)
        b_10 = self.avgpool(feature10)
        # mixed 6a
        x = self.mixed_6a(x)
        c = self.avgpool(x)
        # block 17
        x, feature1 = self.Block17_1(x)
        d_1 = self.avgpool(feature1)
        x, feature2 = self.Block17_2(x)
        d_2 = self.avgpool(feature2)
        x, feature3 = self.Block17_3(x)
        d_3 = self.avgpool(feature3)
        x, feature4 = self.Block17_4(x)
        d_4 = self.avgpool(feature4)
        x, feature5 = self.Block17_5(x)
        d_5 = self.avgpool(feature5)
        x, feature6 = self.Block17_6(x)
        d_6 = self.avgpool(feature6)
        x, feature7 = self.Block17_7(x)
        d_7 = self.avgpool(feature7)
        x, feature8 = self.Block17_8(x)
        d_8 = self.avgpool(feature8)
        x, feature9 = self.Block17_9(x)
        d_9 = self.avgpool(feature9)
        x, feature10 = self.Block17_10(x)
        d_10 = self.avgpool(feature10)
        x, feature11 = self.Block17_11(x)
        d_11 = self.avgpool(feature11)
        x, feature12 = self.Block17_12(x)
        d_12 = self.avgpool(feature12)
        x, feature13 = self.Block17_13(x)
        d_13 = self.avgpool(feature13)
        x, feature14 = self.Block17_14(x)
        d_14 = self.avgpool(feature14)
        x, feature15 = self.Block17_15(x)
        d_15 = self.avgpool(feature15)
        x, feature16 = self.Block17_16(x)
        d_16 = self.avgpool(feature16)
        x, feature17 = self.Block17_17(x)
        d_17 = self.avgpool(feature17)
        x, feature18 = self.Block17_18(x)
        d_18 = self.avgpool(feature18)
        x, feature19 = self.Block17_19(x)
        d_19 = self.avgpool(feature19)
        x, feature20 = self.Block17_20(x)
        d_20 = self.avgpool(feature20)
        # mixed 7a
        x = self.mixed_7a(x)
        e = self.avgpool(x)
        # block 8
        x, feature1 = self.Block8_1(x)
        f_1 = self.avgpool(feature1)
        x, feature2 = self.Block8_2(x)
        f_2 = self.avgpool(feature2)
        x, feature3 = self.Block8_3(x)
        f_3 = self.avgpool(feature3)
        x, feature4 = self.Block8_4(x)
        f_4 = self.avgpool(feature4)
        x, feature5 = self.Block8_5(x)
        f_5 = self.avgpool(feature5)
        x, feature6 = self.Block8_6(x)
        f_6 = self.avgpool(feature6)
        x, feature7 = self.Block8_7(x)
        f_7 = self.avgpool(feature7)
        x, feature8 = self.Block8_8(x)
        f_8 = self.avgpool(feature8)
        x, feature9 = self.Block8_9(x)
        f_9 = self.avgpool(feature9)
        x, feature10 = self.block8(x)
        f_10 = self.avgpool(feature10)

        # x = self.conv2d_7b(x)

        fusion_feature = torch.cat((a, b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8, b_9, b_10, c, d_1, d_2, d_3, d_4, d_5,
                                    d_6, d_7, d_8, d_9, d_10, d_11, d_12, d_13, d_14, d_15, d_16, d_17, d_18, d_19, d_20,
                                    e, f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9, f_10), dim=1)
        # print(fusion_feature.shape)
        # fusion_feature_1_1 = self.avgpool_1a(fusion_feature)
        return fusion_feature #, fusion_feature_1_1

    def logits(self, features):
        x = self.avgpool_1a(features)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        fusion_feature = self.features(input)
        # print(x.shape)
        # x = self.logits(x)
        return fusion_feature #, fusion_1_1


def inceptionresnetv2(num_classes=1000, pretrained='imagenet'):
    r"""InceptionResNetV2 model architecture from the
    `"InceptionV4, Inception-ResNet..." <https://arxiv.org/abs/1602.07261>`_ paper.
    """
    if pretrained:
        settings = pretrained_settings['inceptionresnetv2'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = InceptionResNetV2(num_classes=1001)
        # print(model)
        # model.load_state_dict(model_zoo.load_url(settings['url']))

        # pre_dict = torch.load('inception_resnetv2_modify_pre_model.pth')

        if pretrained == 'imagenet':
            new_last_linear = nn.Linear(1536, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']

        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = InceptionResNetV2(num_classes=num_classes)
    return model

'''
TEST
Run this code with:
```
cd $HOME/pretrained-models.pytorch
python -m pretrainedmodels.inceptionresnetv2
```
'''
if __name__ == '__main__':

    # assert inceptionresnetv2(num_classes=10, pretrained=None)
    # print('success')
    # assert inceptionresnetv2(num_classes=1000, pretrained='imagenet')
    # print('success')
    # assert inceptionresnetv2(num_classes=1001, pretrained='imagenet+background')
    # print('success')
    #
    # # fail
    # assert inceptionresnetv2(num_classes=1000, pretrained='imagenet')

    # print(inceptionresnetv2())
    import torch

    input_data = torch.rand(1, 3, 299, 299)
    model = inceptionresnetv2(pretrained='imagenet')
    # print(model)
    output, fusion_feature = model(input_data)
    print(output.shape)
    print(fusion_feature.shape)