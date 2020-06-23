import numpy as np
import pandas as pd
import random
import os
import argparse
import statistics
import SimpleITK as sitk

from model import KNearestNeighbor
from utils import load_dataset, write_mhd_and_raw

DATA_DIR = '../train'
CASE_LIST_PATH = DATA_DIR + '/case_list.txt'

class nFoldCrossVaridation:
    def __init__(self, n_fold, seed, case_list):
        random.seed(seed)

        self.case_list = case_list             
        self.num_data  = len(self.case_list)
        self.n_fold = n_fold
        assert self.num_data % self.n_fold == 0
        self.step = self.num_data // self.n_fold

        random.shuffle(self.case_list)

        self.d_lis = []
        self.l_lis = []
        for case in self.case_list:
            self.d_lis.append(os.path.join(DATA_DIR, 'Image', case + '.mhd'))
            self.l_lis.append(os.path.join(DATA_DIR, 'Label', case + '.mhd'))

        self.logger()


    def __iter__(self):
        stIdx = 0
        edIdx = self.step

        for _ in range(self.n_fold):
            x_train = self.d_lis[:stIdx] + self.d_lis[edIdx:]
            y_train = self.l_lis[:stIdx] + self.l_lis[edIdx:]
            x_valid = self.d_lis[stIdx:edIdx]
            y_valid = self.l_lis[stIdx:edIdx]

            stIdx += self.step
            edIdx += self.step

            yield x_train, y_train, x_valid, y_valid
    
    def logger(self):
        
        stIdx = 0
        edIdx = self.step

        log = []
        for fold in range(self.n_fold):
            train_case = self.case_list[:stIdx] + self.case_list[edIdx:]
            valid_case = self.case_list[stIdx:edIdx]
            log.append('Fold {}, TRAIN: {}\n'.format(fold+1, train_case))
            log.append('Fold {}, VALID: {}\n'.format(fold+1, valid_case))

            stIdx += self.step
            edIdx += self.step

        with open('./fold_log.txt', mode='w', encoding='UTF-8') as f:
            f.writelines(log)   

def check_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p)

def calc_dice(pred, label):
    pred = pred.astype(bool)
    label = label.astype(bool)

    dice_score = []
    for i in range(pred.shape[0]):
        dice = 2. * (pred[i] & label[i]).sum() / (pred[i].sum() + label[i].sum())
        dice_score.append(dice)

    return statistics.mean(dice_score)

def get_prob_atlas(args, y_train):
    data_len = y_train.shape[0]
    t_label = np.sum(y_train.astype(np.float32), axis=0)
    p_atl = t_label / data_len
    return p_atl

def get_features(args, data, label, case_list):
    for case in case_list:
        data.append(os.path.join(args.feature_path, case))
        label.append(os.path.join(DATA_DIR, 'Label', case))
    return data, label

def save_img(args, pred, test_case_list):
    for i, p in enumerate(test_case_list):
        write_mhd_and_raw(sitk.GetImageFromArray(pred[i, :].reshape(args.img_shape)),
        os.path.join(args.save_path, "fold-{}".format(args.fold), "k-{}".format(args.k), p))
        

def train(args, x_train, y_train, x_valid, y_valid):

    model = KNearestNeighbor(args.k, x_train, y_train, args.dist)

    pred = model.fit(x_valid)

    dice = calc_dice(pred, y_valid)

    print('Dice: {}'.format(dice))

    return pred, dice


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path', type=str, default='./result/')
    parser.add_argument('--max_k', type=int, default=30)
    parser.add_argument('--min_k', type=int, default=1)
    parser.add_argument('--img_shape', type=tuple, default=(1024, 512))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_fold', type=int, default=6)
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--check_filter', type=bool, default=False)
    parser.add_argument('--app_feature', type=bool, default=True)
    parser.add_argument('--feature_path', type=str, default='./features/LevelSet/')
    parser.add_argument('--dist', type=str, default='l2', help=['l1', 'l2'])

    args = parser.parse_args()
    return args

def main(args):

    check_dir(args.save_path)

    with open(CASE_LIST_PATH, 'r') as f:
        case_list = [row.strip() for row in f]

    cross_valid = nFoldCrossVaridation(args.n_fold, args.seed, case_list)

    args.fold = 1

    result = []
    for x_t, y_t, x_v, y_v in cross_valid:
        tl = len(x_t)
        vl = len(x_v)

        print('start {} fold'.format(args.fold))

        x_train_name = [p.split(os.sep)[-1] for p in x_t]
        x_valid_name = [p.split(os.sep)[-1] for p in x_v] 

        if args.app_feature:
            x_t, y_t = get_features(args, x_t, y_t, x_train_name)
            x_v, y_v = get_features(args, x_v, y_v, x_valid_name)

        if args.check_filter:
            x_train, y_train = load_dataset(x_t[tl:], y_t[tl:], args.normalize)
            # x_valid, y_valid = load_dataset(x_v[vl:], y_v[tl:], args.normalize)
            x_valid, y_valid = load_dataset(x_v[:vl], y_v[vl:], args.normalize) # for Level Set

        else:
            x_train, y_train = load_dataset(x_t, y_t, args.normalize)
            x_valid, y_valid = load_dataset(x_v, y_v, args.normalize)

        for k_i in range(args.min_k, args.max_k + 1):
            print('K: {}'.format(k_i))

            args.k = k_i

            pred, score = train(args, x_train, y_train, x_valid, y_valid)

            result.append([args.fold, args.k, score])

            save_img(args, pred, x_valid_name)
        
        args.fold += 1
    
    df = pd.DataFrame(result, columns=['fold', 'k', 'Dice'])
    df.to_csv(os.path.join(args.save_path, 'knn_result.csv'), index=False)
        

if __name__ == '__main__':
    args = arg_parser()
    main(args)