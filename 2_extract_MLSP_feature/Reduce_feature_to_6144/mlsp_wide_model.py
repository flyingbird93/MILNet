import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


# define mlsp network
class mlsp_model(nn.Module):
    def __init__(self, input_channel, middle_channel, out_channel):
        super(mlsp_model, self).__init__()
        self.batch1 = nn.Sequential(nn.Conv2d(input_channel, middle_channel, 1, 1),
                                    nn.BatchNorm2d(middle_channel),
                                    nn.ReLU(inplace=True))
        self.batch2 = nn.Sequential(nn.Conv2d(input_channel, middle_channel, 1, 1),
                                    nn.BatchNorm2d(middle_channel),
                                    nn.ReLU(inplace=True))
        self.batch2_1 = nn.Sequential(nn.Conv2d(middle_channel, middle_channel//2, (1, 3), 1, (0, 1)),
                                      nn.BatchNorm2d(middle_channel//2),
                                      nn.ReLU(True))
        self.batch2_2 = nn.Sequential(nn.Conv2d(middle_channel, middle_channel//2, (3, 1), 1, (1, 0)),
                                      nn.BatchNorm2d(middle_channel//2),
                                      nn.ReLU(inplace=True))
        self.batch3 = nn.Sequential(nn.AvgPool2d(3, 1),
                                    nn.Conv2d(input_channel, middle_channel, 1, 1, 1),
                                    nn.BatchNorm2d(middle_channel),
                                    nn.ReLU(inplace=True))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Linear(middle_channel*3, int(middle_channel*1.5)),
                                 nn.BatchNorm1d(int(middle_channel*1.5)),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.25))
        self.fc2 = nn.Sequential(nn.Linear(int(middle_channel*1.5), int(middle_channel*3/8)),
                                 nn.BatchNorm1d(int(middle_channel*3/8)),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.25))
        self.fc3 = nn.Sequential(nn.Linear(int(middle_channel*3/8), out_channel),
                                 nn.BatchNorm1d(out_channel),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.5))

    # forward
    def forward(self, input):
        feature1 = self.batch1(input)
        feature2 = self.batch2(input)
        feature2_1 = self.batch2_1(feature2)
        feature2_2 = self.batch2_2(feature2)
        feature3 = self.batch3(input)
        fusion_feature = torch.cat((feature1, feature2_1, feature2_2, feature3), dim=1)
        fusion_feature = self.gap(fusion_feature)

        # .view(-1, 2048*3)
        fusion_feature = fusion_feature.squeeze(2).squeeze(2)
        # print(fusion_feature.shape)
        output = self.fc3(self.fc2(self.fc1(fusion_feature)))
        output = F.softmax(output, dim=1)
        # print(output.shape)
        return output


if __name__ == '__main__':
    import torch

    input_data = torch.rand(2, 16928, 5, 5)
    model = mlsp_model(16928, 2048, 10)
    output = model(input_data)
    print(output.shape)
    print(output)