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
from torchvision import transforms


class AVADataset(data.Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, train_file_path, root_dir, transform):
        # train
        self.train_img_id_list = list(pd.read_csv(train_file_path, index_col=False, header=None, sep=' ')[0])
        self.train_img_score_list = list(pd.read_csv(train_file_path, index_col=False, header=None, sep=' ')[1])
        # self.anno_file = pd.read_csv(anno_files, index_col=False, header=None, sep=' ')
        self.root = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.train_img_id_list)

    def __getitem__(self, idx):
        train_feature_path = self.root + str(self.train_img_id_list[idx]) + '.jpg'
        img_id = self.train_img_id_list[idx]

        image = Image.open(train_feature_path)
        image = image.convert('RGB')

        # train score gt
        train_score = self.train_img_score_list[idx]

        # train_anno = self.anno_file[self.anno_file[0] == img_id].iloc[0, 1:11].as_matrix()
        if train_score >= 5:
            train_label = 1
        else:
            train_label = 0

        sample = {'img': image, 'label': train_label, 'id': img_id, 'score': train_score}

        if self.transform:
            sample['img'] = self.transform(sample['img'])

        return sample
