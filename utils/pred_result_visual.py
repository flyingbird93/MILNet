from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

# read pred and gt info
train_feature_refer = pd.read_csv('../data/pred_train_anno_simple.txt', sep=' ', header=None, index_col=False)
test_feature_refer = pd.read_csv('../data/pred_test_anno_simple.txt', sep=' ', header=None, index_col=False)
gt_anno = pd.read_csv('../data/all_anno_new.txt', sep=' ', header=None, index_col=False)
test_anno_gt = pd.read_csv('../data/test_dataset_19928.txt', sep=' ', header=None, index_col=False)
train_anno_gt = pd.read_csv('../data/train_val_dataset_235574.txt', sep=' ', header=None, index_col=False)


# rank score and return N
def rank_score_dist(input_data, gt_data, N):
    input_score = input_data[11].values
    gt_score = gt_data[1].values
    img_id = input_data[0].values
    rank_img_id_list = []

    dist = np.abs(input_score - gt_score)
    rank_index = np.argsort(dist)[-1:-N-1:-1]
    # rank_index = np.argsort(dist)[:N]

    for i in rank_index:
        rank_img_id_list.append(img_id[i])
    return rank_img_id_list


# save img_id
def write_info(input_list, save_path):
    with open(save_path, 'w') as f:
        for i in input_list:
            line = str(i) + '\n'
            f.write(line)


# plot img and anno
def plot_img_pred_gt_anno_score(input_list, pred_anno, gt_anno, img_root, save_root):
    # max score and min score refer
    for i in tqdm(range(len(input_list))):
        fig1 = plt.figure(figsize=(6, 8))
        x = np.arange(1, 11)

        pred_y = pred_anno[pred_anno[0]==input_list[i]].iloc[0, 1:11].values
        pred_score = pred_anno[pred_anno[0]==input_list[i]].iloc[0, 11]

        gt_y = gt_anno[gt_anno[0] == input_list[i]].iloc[0, 1:11].values
        gt_score = gt_anno[gt_anno[0] == input_list[i]].iloc[0, 11]

        ax1 = fig1.add_subplot(211)
        pred_title = 'pred score: ' + str(round(pred_score, 3)) + ' | ' + 'gt_score: ' + str(round(gt_score, 3))
        ax1.set_title(pred_title, fontsize=8, color='b')
        plt.bar(x, pred_y, color='r', alpha=0.5, width=0.4, label='pred')
        plt.bar(x+0.4, gt_y, color='g', alpha=0.5, width=0.4, label='gt')
        plt.legend(loc='upper right')

        # ax1.axis('off')

        # ax2 = fig1.add_subplot(222)
        # gt_y = gt_anno[gt_anno[0]==input_list[i]].iloc[0, 1:11].values
        # gt_score = gt_anno[gt_anno[0]==input_list[i]].iloc[0, 11]
        # gt_title = 'gt score: ' + str(round(gt_score, 3))
        # ax2.set_title(gt_title, fontsize=8, color='b')
        # plt.bar(x, gt_y, color='b')
        # ax2.axis('off')

        ax3 = fig1.add_subplot(212)
        img_data = Image.open(img_root + str(input_list[i]) + '.jpg')
        ax3.set_title('origin image', fontsize=8, color='b')
        ax3.imshow(img_data)
        ax3.axis('off')
        # plt.show()
        save_path = save_root + str(i) + '_' + str(input_list[i])
        fig1.savefig(save_path)

# rank and save list
N = 200
rank_img_list = rank_score_dist(test_feature_refer, test_anno_gt, N)
# rank_img_list = rank_score_dist(train_feature_refer, train_anno_gt, N)
write_info(rank_img_list, 'test_rank_worst_200_img_id.txt')

# save fig
img_root = '/home/flyingbird/1_Data/images/'
# train_list = [150438]
save_path = 'new_train_worst_2000/'
# plot_img_pred_gt_anno_score(rank_img_list, test_feature_refer, gt_anno, img_root, save_path)
# plot_img_pred_gt_anno_score(rank_img_list, train_feature_refer, gt_anno, img_root, save_path)