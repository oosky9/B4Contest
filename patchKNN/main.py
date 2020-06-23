import numpy as np
import pandas as pd
import random
import os
import argparse
import statistics
import SimpleITK as sitk

from model import KNearestNeighbor
from utils import load_dataset, get_patch_dataset, patch_integrator, write_mhd_and_raw

from tqdm import tqdm

DATA_DIR = '../train'
FEATURE_DIR = './features/LevelSet/'
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
            # self.d_lis.append(os.path.join(DATA_DIR, 'Image', case + '.mhd'))
            self.d_lis.append(os.path.join(FEATURE_DIR, case + '.mhd'))
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
        if np.isnan(dice):
            dice = 1.0
        dice_score.append(dice)

    return statistics.mean(dice_score)


def save_img(args, pred, test_case_list):
    print(pred[0].shape)
    for i, p in enumerate(test_case_list):
        write_mhd_and_raw(sitk.GetImageFromArray(pred[i]),
        os.path.join(args.save_path, "fold-{}".format(args.fold), p))
        

def train(args, x_train, y_train, x_valid, y_valid):

    model = KNearestNeighbor(args.k, x_train, y_train, args.dist)

    pred = model.fit(x_valid)

    dice = calc_dice(pred, y_valid)

    return pred, dice


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path', type=str, default='./result/')
    parser.add_argument('--max_k', type=int, default=50)
    parser.add_argument('--min_k', type=int, default=1)
    parser.add_argument('--img_shape', type=tuple, default=(1024, 512))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_fold', type=int, default=6)
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--dist', type=str, default='l2', help=['l2'])
    parser.add_argument('--patch_size', type=int, default=64)

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

        x_train, y_train = load_dataset(x_t, y_t, args.normalize, flat=False)
        x_valid, y_valid = load_dataset(x_v, y_v, args.normalize, flat=False)

        x_patch_train, y_patch_train = get_patch_dataset(x_train, y_train, args.patch_size)
        x_patch_valid, y_patch_valid = get_patch_dataset(x_valid, y_valid, args.patch_size)

        best_k_per_patch = []
        best_patch = []
        for p_idx in range(x_patch_train.shape[0]):
            
            print("No. {}".format(p_idx + 1))

            x_pat_tn, y_pat_tn = x_patch_train[p_idx, :, :], y_patch_train[p_idx, :, :]
            x_pat_vd, y_pat_vd = x_patch_valid[p_idx, :, :], y_patch_valid[p_idx, :, :]

            max_score = 0
            best_k = 0
            best_p = 0
            for k_i in tqdm(range(args.min_k, args.max_k + 1)):

                args.k = k_i

                pred, score = train(args, x_pat_tn, y_pat_tn, x_pat_vd, y_pat_vd)

                result.append([args.fold, p_idx, args.k, score])

                if score > max_score:
                    max_score = score
                    best_k = args.k
                    best_p = pred
                
                if score >= 1:
                    break
            
            print('Dice: {}, Best K: {}'.format(max_score, best_k))
            best_k_per_patch.append('patch No.{}, best k {}\n'.format(str(p_idx + 1), str(best_k)))
            best_patch.append(best_p)
        
        out = patch_integrator(best_patch, args.img_shape, args.patch_size)
        dice_score = calc_dice(np.asarray(out), y_valid)
        print('Dice: {}'.format(dice_score))
        best_k_per_patch.append('dice score {}'.format(dice_score))

        save_img(args, out, x_valid_name)

        with open(os.path.join(args.save_path, 'fold-{}_best_k.txt'.format(args.fold)), mode='w', encoding='UTF-8') as f:
            f.writelines(best_k_per_patch)

        args.fold += 1
    
    df = pd.DataFrame(result, columns=['fold', 'patch No.', 'k', 'Dice'])
    df.to_csv(os.path.join(args.save_path, 'knn_result.csv'), index=False)
        

if __name__ == '__main__':
    args = arg_parser()
    main(args)