import torch
from torch import nn
from torch.nn import functional as F


class Rs_GCN(nn.Module):
    def __init__(self, in_channels, inter_channels, bn_layer=True):
        super(Rs_GCN, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv1d
        max_pool = nn.MaxPool1d
        bn = nn.BatchNorm1d

        self.g = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
                               nn.BatchNorm1d(self.inter_channels),
                               nn.ReLU(inplace=True))

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )

            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W_1_1_pad = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        # self.theta = None
        # self.phi = None

        self.theta = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm1d(self.inter_channels),
                                   nn.ReLU(inplace=True))
        self.phi = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm1d(self.inter_channels),
                                 nn.ReLU(inplace=True))

    def forward(self, v):
        '''
        :param v: (B, D, N)
        :return:
        '''
        batch_size = v.size(0)

        g_v = self.g(v).view(batch_size, self.inter_channels, -1) # b, new_channel, n
        g_v = g_v.permute(0, 2, 1) # b, n, new_channel

        theta_v = self.theta(v).view(batch_size, self.inter_channels, -1) # b, new_channel, n
        theta_v = theta_v.permute(0, 2, 1) # b, n, new_channel
        phi_v = self.phi(v).view(batch_size, self.inter_channels, -1) # b, new_channel, n
        # print('phi_v: ', phi_v.shape)
        R = torch.matmul(theta_v, phi_v) # b, n, n
        N = R.size(-1) # n
        R_div_C = R / N # b, n, n
        # print('r_div_c: ', R_div_C.shape)

        y = torch.matmul(R_div_C, g_v) # b, new_channel, n
        y = y.permute(0, 2, 1).contiguous() # b, n, new_channel11
        y = y.view(batch_size, self.inter_channels, *v.size()[2:]) # b, new_channel, n
        W_y = self.W(y) # b, origin_channel, n
        v_star = W_y + v # b, original_channel, n

        return v_star


if __name__ == '__main__':
    import torch

    input_data = torch.rand(10, 2048, 10)
    model = Rs_GCN(2048, 1024)

    output_star = model(input_data)
    print(output_star.shape)
    # print(R_div_c.shape)
