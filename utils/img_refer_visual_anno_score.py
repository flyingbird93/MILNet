from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

# read pred and gt info
train_best_df = pd.read_csv('train_rank_best_2000_img_id.txt', sep=' ', header=None, index_col=False)
test_best_df = pd.read_csv('test_rank_best_200_img_id.txt', sep=' ', header=None, index_col=False)
test_worst_df = pd.read_csv('test_rank_worst_200_img_id.txt', sep=' ', header=None, index_col=False)
train_worst_df = pd.read_csv('train_rank_worst_2000_img_id.txt', sep=' ', header=None, index_col=False)

train_refer_df = pd.read_csv('train_refer_100.txt', sep=' ', header=None, index_col=False)
test_refer_df = pd.read_csv('test_refer_100.txt', sep=' ', header=None, index_col=False)

gt_anno = pd.read_csv('../data/all_anno_new.txt', sep=' ', header=None, index_col=False)


# plot img and anno
def plot_img_pred_gt_anno_score(input_list, refer_df, gt_anno, img_root, save_root):
    # max score and min score refer
    for i in tqdm(range(len(input_list))):
        fig1 = plt.figure(figsize=(12, 6))
        x = np.arange(1, 11)

        gt_y = gt_anno[gt_anno[0] == input_list[i]].iloc[0, 1:11].values
        gt_score = gt_anno[gt_anno[0] == input_list[i]].iloc[0, 11]

        ax1 = fig1.add_subplot(241)
        img_data = Image.open(img_root + str(input_list[i]) + '.jpg')
        ax1.imshow(img_data)
        ax1.axis('off')
        pred_title = 'gt_score: ' + str(round(gt_score, 3))
        ax1.set_title(pred_title, fontsize=8, color='b')
        # plt.bar(x, pred_y, color='r', alpha=0.5, width=0.4, label='pred')
        ax2 = fig1.add_subplot(245)
        # ax2.axis('off')
        plt.bar(x, gt_y, color='g', alpha=0.5, width=0.4, label='gt')
        plt.ylim(0, 0.5)
        # plt.legend(loc='upper right')

        # refer list
        refer_list = refer_df[refer_df[0]==input_list[i]].iloc[0, 1:4].values

        for j in range(len(refer_list)):
            # ax fig
            index_str1 = 242 + j
            ax = fig1.add_subplot(index_str1)
            refer_data = Image.open(img_root + str(int(refer_list[j])) + '.jpg')
            ax.imshow(refer_data)
            ax.axis('off')

            refer_score = gt_anno[gt_anno[0] == refer_list[j]].iloc[0, 11]
            refer_title = 'gt_score: ' + str(round(refer_score, 3))
            ax.set_title(refer_title, fontsize=8, color='b')

            index_str2 = 246 + j
            ax2 = fig1.add_subplot(index_str2)
            refer_anno = gt_anno[gt_anno[0]==refer_list[j]].iloc[0, 1:11].values
            plt.bar(x, refer_anno, color='g', alpha=0.5, width=0.4, label='gt')
            plt.ylim(0, 0.5)
            # ax2.axis('off')
        # plt.show()
        save_path = save_root + str(i) + '_' + str(input_list[i])
        fig1.savefig(save_path)

img_root = '/home/flyingbird/1_Data/images/'
# test_worst_list = list(test_worst_df[0])
# save_root = 'test_worst_200_refer_anno_score/'
# plot_img_pred_gt_anno_score(test_worst_list, test_refer_df, gt_anno, img_root, save_root)

# test_best
# test_best_list = list(test_best_df[0])
train_best_list = list(train_best_df[0])
save_root = 'train_best_2000_refer_anno_score/'
plot_img_pred_gt_anno_score(train_best_list, train_refer_df, gt_anno, img_root, save_root)