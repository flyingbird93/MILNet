# -*- coding: utf-8 -*-
import os
import numpy as np

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from scipy import stats
from tqdm import tqdm
from config_aesthetic import get_args
from utils.filter_nan import filter_nan

from data.gcn_dataloader_6144 import AVADataset
from model.single_rsgcn_loss_emd import RsgcnModel
from model.adaptive_emd_loss import ada_emd_loss
from model.emd_loss_metric import compute_mse, emd_dis


def main():
    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # config
    config = get_args()

    # model
    model = RsgcnModel(6144, 512, 512, 5, 10)
    model = model.cuda()

    # warm start
    if config.warm_start:
        model.load_state_dict(torch.load(os.path.join(config.ckpt_path,
                                                      'ILG-semantic-GCN-obj-color-loss-ada-EMD-visual-model-epoch-%d.pkl' % config.warm_start_epoch)))
        print('Successfully loaded pretrain model')

    # setting lr
    conv_base_lr = config.conv_base_lr
    optimizer = optim.Adam(model.parameters(), conv_base_lr)

    # loss function
    criterion = ada_emd_loss

    # record training log
    result_dir = config.result_path + 'ILG_semantic_GCN_obj_color_ada_EMD_visual'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    writer = SummaryWriter(log_dir=result_dir)

    # model size
    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    # 　training
    if config.train:
        # read dataset
        trainset = AVADataset(config.train_csv_file, config.refer_img_path, config.train_img_path, config.anno_file, config.train_refer_file)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                                                   shuffle=True, num_workers=config.num_workers)

        # for early stopping
        train_losses = []
        init_C = 0
        init_throshold = 0.2
        alpha = 0.99

        # start training
        print('its learning time: ')

        for epoch in range(config.warm_start_epoch, config.epochs):
            batch_losses = []

            for i, data in tqdm(enumerate(train_loader)):
                refer_feature = data['refer_feature'].to(device).float()
                refer_feature = torch.transpose(refer_feature, 1, 2)
                anno = data['anno'].to(device).float()
                anno = anno.view(-1, 10, 1)

                # 输出分数分布
                gcn_outputs = model(refer_feature)
                gcn_outputs = gcn_outputs.view(-1, 10, 1)

                optimizer.zero_grad()

                # loss function
                loss_gcn = criterion(anno, gcn_outputs, init_C, init_throshold)
                init_C = alpha * loss_gcn.detach() + (1-alpha) * init_C

                batch_losses.append(loss_gcn.item())

                # backward
                loss_gcn.backward()
                optimizer.step()

                if i % 50 == 49:
                    print('Epoch: %d/%d | Step: %d/%d | Training Rank loss: %.4f' % (
                    epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size + 1, loss_gcn.data.item()))

            # update throshold
            init_throshold = torch.mean(torch.Tensor(batch_losses))

            # compute mean loss
            avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size + 1)
            train_losses.append(avg_loss)
            print('Epoch %d averaged training Rank loss: %.4f' % (epoch + 1, avg_loss))
            writer.add_scalars('Loss_group', {'train_loss': avg_loss}, epoch)
            print('Epoch %d gcn loss: %.4f' % (epoch + 1, loss_gcn))
            writer.add_scalars('Loss_group', {'gcn_loss': loss_gcn}, epoch)

            # exponetial learning rate decay
            if (epoch + 1) % 3 == 0:
                conv_base_lr = conv_base_lr / 10
                optimizer = optim.Adam(model.parameters(), conv_base_lr)
            writer.add_scalars('LR', {'learn_rate': conv_base_lr}, epoch)
            # Use early stopping to monitor training
            # print('Saving model...')
            torch.save(model.state_dict(), os.path.join(config.ckpt_path,
                                                        'ILG-semantic-GCN-obj-color-loss-ada-EMD-visual-model-epoch-%d.pkl' % (epoch + 1)))
            print('Done.\n')

    # 　testing
    if config.test:
        model.eval()
        print('its test time: ')

        testset = AVADataset(config.test_csv_file, config.refer_img_path, config.train_img_path, config.anno_file, config.test_refer_file)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=False,
                                                  num_workers=config.num_workers)

        for test_epoch in range(1, config.epochs):
            pred_score = []
            pred_dis_score = []
            gt_score = []
            gt_dis_score = []

            model.load_state_dict(torch.load(os.path.join(config.ckpt_path, 'best_model.pkl')))

            for data in tqdm(test_loader):
                # forward
                refer_feature = data['refer_feature'].to(device).float()
                refer_feature = torch.transpose(refer_feature, 1, 2)
                score = data['score']
                gt_dis = data['anno']

                with torch.no_grad():
                    gcn_outputs = model(refer_feature)

                gcn_outputs = gcn_outputs.view(-1, 10, 1)

                pred_dis_score += list(gcn_outputs.cpu().numpy())
                gt_dis_score += list(gt_dis.cpu().numpy())

                for elem_output in gcn_outputs:
                    predicted_mean = 0.0
                    for i, elem in enumerate(elem_output, 1):
                        predicted_mean += i * elem
                    pred_score.append(predicted_mean.cpu().numpy()[0])
                gt_score += list(score)

            new_pred_score, new_gt_score, new_pred_dist, new_gt_dist = filter_nan(pred_score, gt_score, pred_dis_score, gt_dis_score)
            # plcc
            pred = np.squeeze(np.array(new_pred_score).astype('float64'))
            gt = np.squeeze(np.array(new_gt_score).astype('float64'))
            plcc, _ = stats.pearsonr(pred, gt)
            print('% PLCC of mean: {} | epoch: {}'.format(plcc, test_epoch))

            # ACC
            correct_nums = 0
            for i in range(len(new_pred_score)):
                if (new_pred_score[i] >= 5 and new_gt_score[i] >= 5) or (new_pred_score[i] < 5 and new_gt_score[i] < 5):
                    correct_nums += 1
            acc = correct_nums / len(new_pred_score)
            print('acc is %f | epoch: %d' % (acc, test_epoch))

            # srocc
            srocc_gcn = stats.spearmanr(new_pred_score, new_gt_score)[0]
            print('% gcn SRCC of mean: {} | epoch: {}'.format(srocc_gcn, test_epoch))
            writer.add_scalars('SROCC', {'GCN SROCC': srocc_gcn}, test_epoch)

            # MSE
            pred_label = torch.Tensor(np.array(new_pred_score))
            gt_label = torch.Tensor(np.array(new_gt_score))
            mse_value = compute_mse(pred_label, gt_label)
            print('% MSE value: {} | epoch: {}'.format(mse_value, test_epoch))

            # emd1
            pred_dis = torch.Tensor(np.array(new_pred_dist))
            pred_dis = torch.squeeze(pred_dis, dim=-1)
            gt_dis = torch.Tensor(np.array(new_gt_dist))
            emd1_value = emd_dis(pred_dis, gt_dis)
            print('% emd1 value: {} | epoch: {}'.format(emd1_value, test_epoch))

            # emd2
            emd2_value = emd_dis(pred_dis, gt_dis, dist_r=2)
            print('% emd2 value: {} | epoch: {}'.format(emd2_value, test_epoch))

        writer.close()


if __name__=='__main__':
    main()



