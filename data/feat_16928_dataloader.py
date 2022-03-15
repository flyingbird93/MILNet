# -*- coding: utf-8

import os

import pandas as pd
from PIL import Image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle
import warnings
warnings.filterwarnings('ignore')

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

    def __init__(self, train_file_path, root_dir, anno_file):
        # train
        self.train_img_id_list = list(pd.read_csv(train_file_path, index_col=False, header=None, sep=' ')[0])
        # self.train_img_score_list = list(pd.read_csv(train_file_path, index_col=False, header=None, sep=' ')[1])
        self.root = root_dir
        self.anno = pd.read_csv(anno_file, header=None, sep=' ', index_col=False)
        # self.transform = transform

    def __len__(self):
        return len(self.train_img_id_list)

    def __getitem__(self, idx):
        train_feature_path = self.root + str(self.train_img_id_list[idx])
        img_name = self.train_img_id_list[idx]

        with open(train_feature_path, 'rb') as f:
            feature = pickle.load(f)
        feature = torch.squeeze(torch.Tensor(feature), dim=0)

        # train score gt
        anno_score = self.anno[self.anno[0] == self.train_img_id_list[idx]].iloc[0, 1:11].values
        train_score = self.anno[self.anno[0] == self.train_img_id_list[idx]].iloc[0, 11]
        if train_score >= 5:
            train_label = 1
        else:
            train_label = 0

        sample = {'img': feature, 'score': train_score, 'label':train_label, 'anno': anno_score, 'id': img_name}

        return sample
