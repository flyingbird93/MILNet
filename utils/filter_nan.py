import numpy as np


def filter_nan(pred_label, gt_label, pred_dist, gt_dist):
    new_pred_label = []
    new_gt_label = []
    new_pred_dist = []
    new_gt_dist = []
    nan_list = []
    for i in range(len(pred_label)):
        if np.isnan(np.array(pred_label[i])):
            nan_list.append(i)
        if not np.isnan(pred_label[i]):
            new_pred_label.append(pred_label[i])
            new_gt_label.append(gt_label[i])
            new_pred_dist.append(pred_dist[i])
            new_gt_dist.append(gt_dist[i])
    return new_pred_label, new_gt_label, new_pred_dist, new_gt_dist


if __name__=='__main__':
    c = [1.2, 5.6, np.nan]
    d = [np.nan, 5.7, 4.2]
    a, b = filter_nan(c, d)
    print(a, b)