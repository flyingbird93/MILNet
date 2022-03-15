import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

# train model predict test score
# find score nearest train img id
# create score related refer index


def main():
    def read_train_featrue(train_list, save_path):
        root = '/home/flyingbird/Data/resnet50_2048/'

        all_tensor = torch.empty(0, 2048).cuda()
        for i in tqdm(range(len(train_list))):
            fetaure_path = root + str(train_list[i])
            tem_tensor = torch.Tensor(np.load(fetaure_path, allow_pickle=True)).cuda()
            all_tensor = torch.cat((all_tensor, tem_tensor), dim=0)
        torch.save(all_tensor.cpu(), save_path)
        return all_tensor

    # read train dataset and predict dataset
    def read_dataset(dataset_path):
        # list record range score num
        dataset_df = pd.read_csv(dataset_path, header=None, index_col=False)
        return dataset_df

    # compute train dataset nearest score and find img id
    def compute_nearest_feature(input_data, refer_dataset, refer_df, train_mode, num):
        nearest_score_dict = []
        input_data_id_list = input_data

        for index in tqdm(range(len(input_data_id_list))):
            elem_refer_list = []
            # print(input_data_id_list[index])
            root = '/home/flyingbird/Data/resnet50_2048/'
            input_id = input_data_id_list[index]
            elem_refer_list.append(input_id)

            input_feature = np.load(root + str(input_id), allow_pickle=True)
            one_elem = torch.Tensor(input_feature).cuda()
            input_elem = one_elem.expand(refer_dataset.shape[0], refer_dataset.shape[1]).cuda()

            distance = F.pairwise_distance(refer_dataset, input_elem, p=2)

            if train_mode:
                train_sort_nearest_index = torch.argsort(distance)[1:num + 1]
            else:
                train_sort_nearest_index = torch.argsort(distance)[:num]

            refer_img_list = [refer_df[i] for i in train_sort_nearest_index]
            elem_refer_list += refer_img_list
            nearest_score_dict.append(elem_refer_list)
        return nearest_score_dict

    # save score refer index
    def save_refer_index_to_npy(save_path, score_refer_img_id):
        with open(save_path, 'w') as f:
            for i in score_refer_img_id:
                line = ''
                for img_id in i:
                    line += str(img_id) + ' '
                line += '\n'
                f.write(line)
        print('save_score_refer_npy done!!!')

    # read train and test df
    train_path = '../../data/ILG_train.txt'
    test_path = '../../data/ILG_test.txt'

    train_df = list(read_dataset(train_path))
    test_df = list(read_dataset(test_path))

    # all_train_tensor = torch.load('all_AADB_train_tensor.pth').cuda()
    all_train_tensor = read_train_featrue(train_df, 'all_AADB_train_tensor.pth')
    # print(all_train_tensor.shape)

    refer_num = 100
    train_refer_dict = compute_nearest_feature(train_df, all_train_tensor, train_df, True, refer_num)
    test_refer_dict = compute_nearest_feature(test_df, all_train_tensor, train_df, False, refer_num)

    # concate train and test dict
    # all_refer_dict = train_refer_dict.update(test_refer_dict)

    # save refer index
    train_save_path_name = 'ILG_train_refer_100.txt'
    save_refer_index_to_npy(train_save_path_name, train_refer_dict)
    test_save_path_name = 'ILG_test_refer_100.txt'
    save_refer_index_to_npy(test_save_path_name, test_refer_dict)


if __name__ == '__main__':
    main()

