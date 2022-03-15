# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

import torchvision.transforms as transforms
import torchvision.models as models
from tensorboardX import SummaryWriter

from scipy import stats
from tqdm import tqdm
import argparse

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from dataloader import AVADataset
from tqdm import tqdm


def main(config):
    # 使用cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    train_transform = transforms.Compose([
        # transforms.
        transforms.ToTensor()])

    # *****model*****
    model = torchvision.models.resnet50(pretrained=True)
    model = model.cuda()

    def my_collate(batch):
        data = [item['img'] for item in batch]
        name = [item['id'] for item in batch]
        return [data, name]

    # 计算训练参数数量
    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    # 　测试模式
    if config.test:
        model.eval()
        print('It`s feature extract time: ')

        # *****data*****
        testset = AVADataset(config.test_csv_file, config.train_img_path, train_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.train_batch_size,
                                                  shuffle=False, num_workers=config.num_workers,
                                                  collate_fn=my_collate)

        for data in tqdm(test_loader):
            # forward
            img = data[0]
            name = data[1]

            for index in range(len(img)):
                with torch.no_grad():
                    input = img[index].unsqueeze(0).to(device).float()
                    _, outputs = model(input)

                outputs = outputs.cpu().numpy()
                # print(outputs.shape)
                test_list_save_imdb = '/home/flyingbird/Data/resnet50_2048/' + str(int(name[index]))
                torch.save(outputs, test_list_save_imdb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--train_img_path', type=str, default='/home/flyingbird/Data/images/')
    parser.add_argument('--val_img_path', type=str, default='/home/flyingbird/Data/images/')
    parser.add_argument('--test_img_path', type=str, default='/home/flyingbird/Data/images/')

    parser.add_argument('--train_csv_file', type=str, default='../data/train_val_dataset.txt')
    parser.add_argument('--val_csv_file', type=str, default='../val.txt')
    parser.add_argument('--test_csv_file', type=str, default='../data/test_dataset_19929.txt')
    parser.add_argument('--refer_csv_file', type=str, default='refer_id_31.txt')

    # training parameters`
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--conv_base_lr', type=float, default=3e-4)
    parser.add_argument('--dense_lr', type=float, default=3e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--test_epoch', type=int, default=40)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='.')
    parser.add_argument('--warm_start', type=bool, default=True)
    parser.add_argument('--warm_start_epoch', type=int, default=0)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--save_fig', type=bool, default=False)

    config = parser.parse_args()

    main(config)

