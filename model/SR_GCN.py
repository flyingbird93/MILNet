import torch
from torch import nn
import numpy as np
import math
from torch.nn import functional as F
# from gmm import GaussianMixture


class SR_GCN(nn.Module):
    def __init__(self, in_channels, inter_channels, adj_op='rand', init='xavier'):
        """SR-GCN module
        param:
              - in_channels: input feat dim
              - inter_channels: middle feat dim
        input:
              - input: channel nums with size of (b, d, n) (batch, 5, 6144)
        output:
              - x: refined feature with size of (b, d, n) (batch, 5, 6144)
        """
        super(SR_GCN, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.adj_op = adj_op
        self.init = init

        self.conv_nd = nn.Conv1d
        self.bn = nn.BatchNorm1d

        self.g = nn.Sequential(self.conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
                               nn.BatchNorm1d(self.inter_channels),
                               nn.ReLU(inplace=True))
        self.theta = nn.Sequential(self.conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm1d(self.inter_channels),
                                   nn.ReLU(inplace=True))
        self.phi = nn.Sequential(self.conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm1d(self.inter_channels),
                                 nn.ReLU(inplace=True))
        self.W = nn.Sequential(self.conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                               self.bn(self.in_channels))

    def reset_parameters(self, init):
        if init == 'uniform':
            stdv = 1. / math.sqrt(self.W.weight.size(1))
            nn.init.uniform_(self.g.weight, -stdv, stdv)
            nn.init.uniform_(self.theta.weight, -stdv, stdv)
            nn.init.uniform_(self.phi.weight, -stdv, stdv)
            nn.init.uniform_(self.W.weight, -stdv, stdv)
        elif init == 'xavier':
            nn.init.xavier_uniform_(self.g.weight)
            nn.init.xavier_uniform_(self.theta.weight)
            nn.init.xavier_uniform_(self.phi.weight)
            nn.init.xavier_uniform_(self.W.weight)

    def forward(self, v):
        '''
        :param v: (B, D, N) bn, 512, 5

        # input feat (bn, 1, 6144)
        # refer feat (bn, 4, 6144)

        :return:
        '''
        batch_size, n, dim = v.shape # bn, 512, 5
        g_v = self.g(v)   # bn, n, new_channel (bn, 512, 5)

        theta_v = self.theta(v) # bn, new_channel, n # (bn, 512, 5)
        phi_v = self.phi(v) # b, new_channel, n # (bn, 512, 5)

        # adj
        if self.adj_op == 'rand':
            dis = torch.rand(batch_size, dim, dim).cuda()  # adj (bn, 5, 5)
        elif self.adj_op == 'color':
            dis = color_transfer_MKL(theta_v, phi_v)  # adj (bn, 5, 5)
        #
        elif self.adj_op == 'semantic':
            dis = torch.matmul(theta_v.permute(0, 2, 1), phi_v) # adj (bn, 5, 5)

        elif self.adj_op == 'edge':
            dis = gmm(theta_v, phi_v, 2, 5).squeeze(0) # 2, 5, 5

        elif self.adj_op == 'fusion':
            color_dis = color_transfer_MKL(theta_v, phi_v)
            semantic_dis = torch.matmul(theta_v.permute(0, 2, 1), phi_v)
            # edge_dis = gmm(theta_v, phi_v, 2, 5).squeeze(0)
            dis = color_dis.cuda() + semantic_dis.cuda() #+ edge_dis.cuda()
        else:
            dis = torch.matmul(theta_v, phi_v) # adj (bn, 5, 5)

        dis = torch.clamp(dis, -100, 100).cuda()
        y = torch.matmul(g_v, dis) # b, 5, 5
        W_y = self.W(y) # b, 512, 5
        v_star = W_y + v # b, 512, 5
        return v_star


def color_transfer_MKL(source, target):
    T_init = torch.empty((0, 5, 5))
    for i in range(source.shape[0]):
        # source.requires_grad = False
        A_index = source[i].detach().cpu().numpy()
        # target.requires_grad = False
        B_index = target[i].detach().cpu().numpy()
        # B_index.requires_grad = False
        A = np.cov(A_index, rowvar=False)
        B = np.cov(B_index, rowvar=False)
        T = MKL(A, B)
        T = T.astype(np.float32)
        T = torch.Tensor(T).unsqueeze(0)
        T_init = torch.cat((T_init, T), dim=0)
    return T_init


def MKL(A, B):

    """
    input feat (bn, dim)
    output feat (dim, dim)
    """

    EPS = 2.2204e-16
    # Da2, Ua = torch.linalg.eig(A, eigenvectors=True)   # 计算特征值和特征向量
    Da2, Ua = np.linalg.eig(A)

    Da2 = np.diag(Da2)           # 方阵对角线元素
    Da2[Da2 < 0] = 0
    Da = np.sqrt(Da2 + EPS)
    C = Da @ np.transpose(Ua) @ B @ Ua @ Da
    Dc2, Uc = np.linalg.eig(C)

    Dc2 = np.diag(Dc2)
    Dc2[Dc2 < 0] = 0
    Dc = np.sqrt(Dc2 + EPS)
    Da_inv = np.diag(1 / (np.diag(Da)))
    T = Ua @ Da_inv @ Uc @ Dc @ np.transpose(Uc) @ Da_inv @ np.transpose(Ua)
    return T


# def gmm(data1, data2, n_components, d):
#     gmm_T_init = torch.empty((0, 5, 5))
#     model = GaussianMixture(n_components, d).cuda()
#     for i in range(len(data1)):
#         data1_elem = data1[i] # (512, 5)
#         # print(data1_elem.shape)
#         data2_elem = data2[i]
#         # model = GaussianMixture(n_components, d)
#         mu, var1 = model.fit(data1_elem, n_iter=1) # var (1, n_components, 5, 5)
#         # print(var1.shape)
#         mu, var2 = model.fit(data2_elem, n_iter=1)
#         var1 = var1.squeeze(0)  # (n_components, 5, 5)
#         var1 = torch.where(torch.isnan(var1), torch.full_like(var1, 0), var1)
#         var2 = var2.squeeze(0)
#         var2 = torch.where(torch.isnan(var2), torch.full_like(var2, 0), var2)
#         sum_comp_dis = torch.zeros((5, 5))
#         for j in range(len(var1)):
#             var1_elem = var1[j] # (5,5)
#             var2_elem = var2[j]
#             dis = compute_gmm_dis(var1_elem, var2_elem)
#             # print(torch.Tensor(dis).shape)
#             sum_comp_dis += torch.Tensor(dis) # 5, 5
#         # print(sum_comp_dis)
#         gmm_T_init = torch.cat((gmm_T_init, sum_comp_dis.unsqueeze(0)), dim=0) # bn, 5, 5
#     # print(gmm_T_init.shape)
#     return gmm_T_init


def compute_gmm_dis(source_var, target_var):
    source_var = source_var.detach().cpu().numpy()
    target_var = target_var.detach().cpu().numpy()
    wasserstein_dis = MKL(source_var, target_var)
    return wasserstein_dis


if __name__ == '__main__':
    import torch
    import time

    start_time = time.time()
    input_data = torch.rand(10, 6144, 5).cuda()
    model = SR_GCN(6144, 512, adj_op='fusion')
    model.cuda()

    output_star = model(input_data)
    print(output_star.shape)
    print(time.time() - start_time)
    # print(R_div_c.shape)
