# -*- coding: utf-8 -*-
import os
import numpy as np

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from scipy import stats
from tqdm import tqdm
import argparse

from data.feat_16928_dataloader import AVADataset
from mlsp_wide_model import mlsp_model
from model.emd_loss_metric import emd_loss_func, compute_mse, emd_dis


def main(config):
    # 使用cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取NIMA模型
    model = mlsp_model(16928, 2048, 10)
    model = model.cuda()

    # 继续上一步训练
    if config.warm_start:
        model.load_state_dict(torch.load(os.path.join(config.ckpt_path,
                                                      'mlsp-baseline-epoch-%d.pkl' % config.warm_start_epoch)))
        print('Successfully loaded pretrain model')

    # 分别设置学习率
    conv_base_lr = config.conv_base_lr
    optimizer = optim.Adam(model.parameters(), conv_base_lr)

    # 两个损失函数 emd——loss 和 交叉熵分类
    criterion1 = emd_loss_func

    # 记录关键参数
    result_dir = config.result_path
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    writer = SummaryWriter(log_dir=result_dir)

    # 计算训练参数数量
    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    # 　训练模式
    if config.train:
        # 读取数据
        trainset = AVADataset(config.train_csv_file, config.train_img_path, config.anno_file)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                                                   shuffle=True, num_workers=config.num_workers)

        # for early stopping
        train_losses = []

        # 开始训练
        print('its learning time: ')

        for epoch in range(config.warm_start_epoch, config.epochs):
            batch_losses = []

            for i, data in tqdm(enumerate(train_loader)):
                refer_feature = data['img'].to(device).float()
                # score = data['score'].to(device).float()
                anno = data['anno'].to(device).float()
                anno = anno.view(-1, 10, 1)
                # score = score.view(-1, 1)

                # 输出分数分布
                gcn_outputs = model(refer_feature)
                gcn_outputs = gcn_outputs.view(-1, 10, 1)

                optimizer.zero_grad()

                # loss function
                loss_gcn = criterion1(anno, gcn_outputs)

                batch_losses.append(loss_gcn.item())

                # backward
                loss_gcn.backward()
                optimizer.step()

                if i % 50 == 49:
                    print('Epoch: %d/%d | Step: %d/%d | Training Rank loss: %.4f' % (
                    epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size + 1, loss_gcn.data.item()))

            # 计算平均损失
            avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size + 1)
            train_losses.append(avg_loss)
            print('Epoch %d averaged training Rank loss: %.4f' % (epoch + 1, avg_loss))
            writer.add_scalars('Loss_group', {'train_loss': avg_loss}, epoch)
            print('Epoch %d gcn loss: %.4f' % (epoch + 1, loss_gcn))
            writer.add_scalars('Loss_group', {'gcn_loss': loss_gcn}, epoch)

            # exponetial learning rate decay　学习率调整
            if (epoch + 1) % 3 == 0:
                conv_base_lr = conv_base_lr / 10
                optimizer = optim.Adam(model.parameters(), conv_base_lr)
            writer.add_scalars('LR', {'learn_rate': conv_base_lr}, epoch)
            # Use early stopping to monitor training
            print('Saving model...')
            torch.save(model.state_dict(), os.path.join(config.ckpt_path,
                                                        'mlsp-baseline-epoch-%d.pkl' % (epoch + 1)))
            print('Done.\n')

    # 　测试模式
    if config.test:
        model.eval()

        print('its test time: ')

        testset = AVADataset(config.test_csv_file, config.train_img_path, config.anno_file)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=False,
                                                  num_workers=config.num_workers)

        for test_epoch in range(1, 5):
            gcn_label = []
            gcn_dis_score = []
            gt_label = []
            gt_dis_score = []

            model.load_state_dict(torch.load(os.path.join(config.ckpt_path,
                                                          'mlsp-baseline-epoch-%d.pkl' % test_epoch)))

            for data in tqdm(test_loader):
                # forward
                refer_feature = data['img'].to(device).float()
                refer_feature = torch.transpose(refer_feature, 1, 2)
                score = data['score']
                gt_dis = data['anno']
                id = data['id']
                # refer_id = data['refer_id']

                with torch.no_grad():
                    gcn_outputs = model(refer_feature)

                gcn_outputs = gcn_outputs.view(-1, 10, 1)

                gcn_dis_score += list(gcn_outputs.cpu().numpy())
                gt_dis_score += list(gt_dis.cpu().numpy())

                for elem_output in gcn_outputs:
                    predicted_mean = 0.0
                    for i, elem in enumerate(elem_output, 1):
                        predicted_mean += i * elem
                    gcn_label.append(predicted_mean.cpu().numpy())
                gt_label += list(score)
                # img_id_list += list(id)
                # refer_id_list = torch.cat((refer_id_list, refer_id), dim=0)

            new_gcn_label = []
            new_gt_label = []
            new_gcn_dis = []
            new_gt_dis = []
            nan_list = []
            for i in range(len(gcn_label)):
                if np.isnan(np.array(gcn_label[i])):
                    nan_list.append(i)
                if not np.isnan(np.array(gcn_label[i])):
                    new_gcn_label.append(gcn_label[i])
                    new_gt_label.append(gt_label[i])

                    new_gcn_dis.append(gcn_dis_score[i])
                    new_gt_dis.append(gt_dis_score[i])

            print(len(new_gcn_label))
            # plcc
            pred = np.squeeze(np.array(new_gcn_label).astype('float64'))
            gt = np.squeeze(np.array(new_gt_label).astype('float64'))
            plcc, _ = stats.pearsonr(pred, gt)
            print('% PLCC of mean: {} | epoch: {}'.format(plcc, test_epoch))

            # ACC
            correct_nums = 0
            for i in range(len(new_gcn_label)):
                if (new_gcn_label[i] >= 5 and new_gt_label[i] >= 5) or (new_gcn_label[i] < 5 and new_gt_label[i] < 5):
                    correct_nums += 1
            acc = correct_nums / len(new_gcn_label)
            print('acc is %f | epoch: %d' % (acc, test_epoch))

            # gcn_srocc
            srocc_gcn = stats.spearmanr(new_gcn_label, new_gt_label)[0]
            print('% gcn SRCC of mean: {} | epoch: {}'.format(srocc_gcn, test_epoch))
            writer.add_scalars('SROCC', {'GCN SROCC': srocc_gcn}, test_epoch)

            # MSE
            pred_label = torch.Tensor(np.array(new_gcn_label))
            gt_label = torch.unsqueeze(torch.Tensor(np.array(new_gt_label)), dim=-1)
            # print(pred_label.size())
            # print(gt_label.size())
            mse_value = compute_mse(pred_label, gt_label)
            print('% MSE value: {} | epoch: {}'.format(mse_value, test_epoch))

            # emd1
            pred_dis = torch.Tensor(np.array(new_gcn_dis))
            pred_dis = torch.squeeze(pred_dis, dim=-1)
            gt_dis = torch.Tensor(np.array(new_gt_dis))
            # print(pred_dis.shape)
            # print(gt_dis.shape)
            emd1_value = emd_dis(pred_dis, gt_dis)
            print('% emd1 value: {} | epoch: {}'.format(emd1_value, test_epoch))

            # emd2
            emd2_value = emd_dis(pred_dis, gt_dis, dist_r = 2)
            print('% emd2 value: {} | epoch: {}'.format(emd2_value, test_epoch))

            # with open('baseline_pred_test_score.txt', 'w') as f:
            #     for i in range(len((new_gcn_label))):
            #         line = str(int(img_id_list[i])) + ' ' + str(float(new_gcn_label[i])) + ' ' + str(float(new_gt_label[i])) + '\n'
            #         f.write(line)
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    # parser.add_argument('--train_img_path', type=str, default='/home/flyingbird/1_Data/feature_6144/')
    # parser.add_argument('--refer_img_path', type=str, default='/home/flyingbird/1_Data/feature_6144/')
    # parser.add_argument('--test_img_path', type=str, default='/home/flyingbird/1_Data/feature_6144/')
    parser.add_argument('--train_img_path', type=str, default='G:/feature_16928_5_5/')
    parser.add_argument('--test_img_path', type=str, default='G:/feature_16928_5_5/')
    # parser.add_argument('--refer_img_path', type=str, default='D:/feature_new_6144/')

    parser.add_argument('--train_csv_file', type=str, default='../data/train_val_dataset_235574.txt')
    parser.add_argument('--train_refer_file', type=str, default='../data/train_refer_100.txt')
    parser.add_argument('--test_csv_file', type=str, default='../data/test_dataset_19928.txt')
    parser.add_argument('--test_refer_file', type=str, default='../data/test_refer_100.txt')

    parser.add_argument('--anno_file', type=str, default='../data/all_anno_new.txt')

    # training parameters`
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--conv_base_lr', type=float, default=3e-5)
    parser.add_argument('--dense_lr', type=float, default=3e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--test_epoch', type=int, default=80)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpt/baseline_2048/')
    parser.add_argument('--result_path', type=str, default='result/baseline_2048/')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default=0)
    parser.add_argument('--warm_start', type=bool, default=True)
    parser.add_argument('--warm_start_epoch', type=int, default=4)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--save_fig', type=bool, default=False)

    config = parser.parse_args()

    main(config)

