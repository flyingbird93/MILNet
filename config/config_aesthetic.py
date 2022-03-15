import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--refer_img_path', type=str, default='D:/ILG_6144/')
    parser.add_argument('--test_img_path', type=str, default='D:/ILG_6144/')
    parser.add_argument('--train_img_path', type=str, default='D:/ILG_6144/')

    parser.add_argument('--train_csv_file', type=str, default='data/ILG_train.txt')
    parser.add_argument('--train_refer_file', type=str, default='data/ILG_train_refer_100.txt')
    parser.add_argument('--test_csv_file', type=str, default='data/ILG_test.txt')
    parser.add_argument('--test_refer_file', type=str, default='data/ILG_test_refer_100.txt')

    parser.add_argument('--anno_file', type=str, default='data/all_anno_new.txt')

    # training parameters`
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--conv_base_lr', type=float, default=3e-4)
    parser.add_argument('--dense_lr', type=float, default=3e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpt/')
    parser.add_argument('--result_path', type=str, default='result/')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default=0)
    parser.add_argument('--warm_start', type=bool, default=False)
    parser.add_argument('--warm_start_epoch', type=int, default=0)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--save_fig', type=bool, default=False)

    config = parser.parse_args()
    return config
# if __name__ == '__main__':
#     config = config()