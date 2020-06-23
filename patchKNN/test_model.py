import numpy as np
import SimpleITK as sitk

import os
import re
import glob
import argparse

from tqdm import tqdm

from model import KNearestNeighbor
from utils import load_dataset, load_test_dataset, get_patch_dataset, get_test_patch_dataset, patch_integrator, write_mhd_and_raw

DATA_DIR = '../train'
FEATURE_DIR = './features/Max/'
CASE_LIST_PATH = DATA_DIR + '/case_list.txt'

TEST_DATA_DIR = '../test'
CASE_TEST_LIST_PATH = TEST_DATA_DIR + '/case_list.txt'

def get_matching_list(l):
    regix = re.compile('\d+')

    matching = []
    for string in l:
        matching.append(int(regix.findall(string)[-1]))
    
    matching = matching[:-1]
    return matching


def get_best_k(args, p):
    file_list  = glob.glob(os.path.join(p, '*best_k.txt'))

    best_k_of_each_fold = []
    for f_i in file_list:
        with open(f_i, mode='r') as f:
            l = f.readlines()
            m = get_matching_list(l)
            best_k_of_each_fold.append(m)
    
    num_arr = np.asarray(best_k_of_each_fold)
    
    best_k = list(np.round(np.mean(num_arr, axis=0)).astype(np.int64))
    
    with open(os.path.join(args.save_model_path, 'model.txt'), mode='w', encoding='UTF-8') as f:
        f.writelines(str(best_k))
    
    return best_k

def save_img(args, pred, test_case_list):
    for i, p in enumerate(test_case_list):
        write_mhd_and_raw(sitk.GetImageFromArray(pred[i]),
        os.path.join(args.output_path, p))
        

def check_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p)
    
def test(args, x_train, y_train, x_test):

    model = KNearestNeighbor(args.k, x_train, y_train, args.dist)

    pred = model.fit(x_test)

    return pred

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_best_k_path', type=str, default='./result/')
    parser.add_argument('--output_path', type=str, default='./output/')
    parser.add_argument('--save_model_path', type=str, default='./model/')
    parser.add_argument('--img_shape', type=tuple, default=(1024, 512))
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--dist', type=str, default='l2', help=['l2'])
    parser.add_argument('--patch_size', type=int, default=64)

    args = parser.parse_args()
    return args

def main(args):

    check_dir(args.save_model_path)
    check_dir(args.output_path)

    model_k_list = get_best_k(args, args.save_best_k_path)

    with open(CASE_LIST_PATH, 'r') as f:
        case_list = [row.strip() for row in f]

    with open(CASE_TEST_LIST_PATH, 'r') as f:
        case_test_list = [row.strip() for row in f]

    d_lis = []
    l_lis = []
    for case in case_list:
        d_lis.append(os.path.join(DATA_DIR, 'Image', case + '.mhd'))
        l_lis.append(os.path.join(DATA_DIR, 'Label', case + '.mhd'))
    
    t_lis = []
    for case in case_test_list:
        t_lis.append(os.path.join(TEST_DATA_DIR, 'Image', case + '.mhd'))

    x_train, y_train = load_dataset(d_lis, l_lis, args.normalize, flat=False)
    x_patch_train, y_patch_train = get_patch_dataset(x_train, y_train, args.patch_size)

    x_test = load_test_dataset(t_lis, args.normalize, flat=False)
    x_patch_test = get_test_patch_dataset(x_test, args.patch_size)

    pred_patch = []
    for p_idx, k in enumerate(tqdm(model_k_list)):

        args.k = k

        x_pat_tn, y_pat_tn = x_patch_train[p_idx, :, :], y_patch_train[p_idx, :, :]
        x_pat_vd = x_patch_test[p_idx, :, :]

        pred_patch.append(test(args, x_pat_tn, y_pat_tn, x_pat_vd))
    
    out = patch_integrator(pred_patch, args.img_shape, args.patch_size)

    output_path_list = [p + '.mhd' for p in case_test_list]
    save_img(args, out, output_path_list)


if __name__ == '__main__':
    args = arg_parser()
    
    main(args)