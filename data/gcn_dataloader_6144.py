# -*- coding: utf-8

import os
import pandas as pd
import pickle
from torch.utils import data
import numpy as np
import torch


class AVADataset(data.Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, train_file_path, refer_path, root_dir, anno_file, refer_file):
        # train
        self.train_img_id_list = list(pd.read_csv(train_file_path, index_col=False, header=None, sep=' ')[0])
        self.refer_root = refer_path
        self.root = root_dir
        self.anno = pd.read_csv(anno_file, header=None, sep=' ', index_col=False)
        self.refer = pd.read_csv(refer_file, header=None, sep=' ', index_col=False)
        # self.data_info = pd.read_csv(data_info, sep=',', index_col=False)

    def __len__(self):
        return len(self.train_img_id_list)

    def __getitem__(self, idx):
        train_feature_path = self.root + str(self.train_img_id_list[idx]) # 1, 16928, 5, 5
        with open(train_feature_path, 'rb') as f:
            train_original_feature = pickle.load(f)
        img_id = int(self.train_img_id_list[idx])

        refer_feature_path = self.refer_root + str(self.train_img_id_list[idx])
        with open(refer_feature_path, 'rb') as f:
            refer_fusion_feature = pickle.load(f)
        # print(refer_fusion_feature.shape)
        # refer_fusion_feature = np.load(refer_feature_path, allow_pickle=True)
        # refer_fusion_feature = np.expand_dims(refer_fusion_feature, axis=0)

        # refer feature
        # print(self.refer[self.refer[0] == self.train_img_id_list[idx]])
        refer_id_list = self.refer[self.refer[0] == self.train_img_id_list[idx]].iloc[0, 1:].values
        refer_anno = torch.Tensor(self.anno[self.anno[0] == self.train_img_id_list[idx]].iloc[0, 1:].values).unsqueeze(0)

        for i in range(4):
            # refer_id_list:
            refer_feature_path = self.refer_root + str(int(refer_id_list[i]))
            with open(refer_feature_path, 'rb') as f:
                refer_feature = pickle.load(f)
            # print(refer_feature.shape)
            # refer_feature = np.expand_dims(refer_feature, axis=0)
            # feature concate
            refer_fusion_feature = np.concatenate((refer_fusion_feature, refer_feature), axis=0) # 7, 1, 16928, 1, 1

        # refer_fusion_feature = np.squeezerefer_fusion_feature)
        # print(refer_fusion_feature.shape)
        anno_score = self.anno[self.anno[0] == self.train_img_id_list[idx]].iloc[0, 1:11].values
        train_score = self.anno[self.anno[0] == self.train_img_id_list[idx]].iloc[0, 11]
        if train_score >= 5:
            train_label = 1
        else:
            train_label = 0

        sample = {'train_feature': train_original_feature, 'refer_feature': refer_fusion_feature, 'refer_anno': refer_anno, 'id': img_id,
                  'score': train_score, 'label': train_label, 'anno': anno_score}

        return sample
