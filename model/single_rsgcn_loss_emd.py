import torch.nn as nn
from Rs_GCN import Rs_GCN
from model.SR_GCN import SR_GCN
import torch.nn.functional as F
import torch


# define mlsp network
class RsgcnModel(nn.Module):
    def __init__(self, input_channel, gcn_feature, reduce_gcn_feature, gcn_node_num, out_emd):
        super(RsgcnModel, self).__init__()
        # reduce
        self.refer_reduce_fc = nn.Sequential(nn.Conv1d(input_channel, gcn_feature, 1),
                                             nn.BatchNorm1d(gcn_feature),
                                             nn.ReLU(inplace=True))

        # gcn
        self.gcn_nodes = gcn_node_num
        self.gcn1 = Rs_GCN(gcn_feature, reduce_gcn_feature)
        self.gcn2 = Rs_GCN(gcn_feature, reduce_gcn_feature)
        self.gcn3 = Rs_GCN(gcn_feature, reduce_gcn_feature)
        self.gcn4 = Rs_GCN(gcn_feature, reduce_gcn_feature)
        self.gcn5 = Rs_GCN(gcn_feature, reduce_gcn_feature)
        self.gcn6 = Rs_GCN(gcn_feature, reduce_gcn_feature)

        # self.gcn1 = SR_GCN(gcn_feature, reduce_gcn_feature, adj_op='semantic')
        # self.gcn2 = SR_GCN(gcn_feature, reduce_gcn_feature, adj_op='semantic')
        # self.gcn3 = SR_GCN(gcn_feature, reduce_gcn_feature, adj_op='semantic')
        # self.gcn4 = SR_GCN(gcn_feature, reduce_gcn_feature, adj_op='semantic')
        # self.gcn5 = SR_GCN(gcn_feature, reduce_gcn_feature, adj_op='semantic')
        # self.gcn6 = SR_GCN(gcn_feature, reduce_gcn_feature, adj_op='semantic')

        self.gcn_fc = nn.Linear(gcn_feature*self.gcn_nodes, out_emd)

        # output 10 scores
        self.fusion_feature_channel = gcn_feature*self.gcn_nodes # + 2048*3
        self.fc = nn.Linear(self.fusion_feature_channel, out_emd)

    # forward
    def forward(self, refer):
        # reduce train and refer
        refer_size = refer.shape
        refer_reduce_feature = self.refer_reduce_fc(refer)

        # gcn
        gcn_feature1 = self.gcn1(refer_reduce_feature)
        gcn_feature2 = self.gcn2(gcn_feature1)
        gcn_feature3 = self.gcn3(gcn_feature2)
        gcn_feature4 = self.gcn4(gcn_feature3)
        gcn_feature5 = self.gcn5(gcn_feature4)
        gcn_feature6 = self.gcn6(gcn_feature5)
        gcn_feature6 = gcn_feature6.contiguous()

        gcn_size = gcn_feature6.shape
        gcn_feature = gcn_feature6.view(gcn_size[0], gcn_size[1] * gcn_size[2])

        # same gt
        gcn_fc_output = self.fc(gcn_feature)
        gcn_fc_output = F.softmax(gcn_fc_output, dim=1)
        return gcn_fc_output


if __name__ == '__main__':
    import torch
    import numpy as np

    input_data_refer = torch.rand(64, 5, 6144).cuda()
    input_data_refer = torch.transpose(input_data_refer, 1, 2)

    model = RsgcnModel(6144, 512, 512, 5, 10).cuda()

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    gcn_output = model(input_data_refer)
    print(gcn_output.shape)
