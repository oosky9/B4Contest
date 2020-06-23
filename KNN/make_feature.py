import SimpleITK as sitk
import numpy as np

from utils import load_dataset, load_sitk_dataset, write_mhd_and_raw
from filters import GaussianFilter, MaxFilter, LevelSet
import os
import argparse

DATA_DIR = '../train'
CASE_LIST_PATH = DATA_DIR + '/case_list.txt'

def check_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p)

def filters(sitk_img, filter=None):

    if filter == 'LoG':
        out = sitk.LaplacianRecursiveGaussianImageFilter().Execute(sitk_img)
        return out
    elif filter == 'Sigmoid':
        out = sitk.SigmoidImageFilter().Execute(sitk_img)
        return out
    elif filter == 'LaplacianSharp':
        out = sitk.LaplacianSharpeningImageFilter().Execute(sitk_img)
        return out
    elif filter == 'SmoothGauss':
        out = sitk.SmoothingRecursiveGaussianImageFilter().Execute(sitk_img)
        return out
    elif filter == 'BlackTop':
        out = sitk.BlackTopHatImageFilter().Execute(sitk_img)
        return out
    elif filter == 'WhiteTop':
        out = sitk.WhiteTopHatImageFilter().Execute(sitk_img)
        return out
    elif filter == 'Median':
        out = sitk.MedianImageFilter().Execute(sitk_img)
        return out
    elif filter == 'Mean':
        out = sitk.MeanImageFilter().Execute(sitk_img)
        return out
    elif filter == 'RecGauss':
        out = sitk.RecursiveGaussianImageFilter().Execute(sitk_img)
        return out
    elif filter == 'DiscGauss':
        out = sitk.DiscreteGaussianImageFilter().Execute(sitk_img)
        return out
    elif filter == 'Gauss':
        np_img = sitk.GetArrayFromImage(sitk_img)
        out = GaussianFilter().Execute(np_img)
        out = sitk.GetImageFromArray(out)
        return out
    elif filter == 'Max':
        np_img = sitk.GetArrayFromImage(sitk_img)
        out = MaxFilter().Execute(np_img)
        out = sitk.GetImageFromArray(out)
        return out
    elif filter == 'GaussToMax':
        np_img = sitk.GetArrayFromImage(sitk_img)
        out = GaussianFilter().Execute(np_img)
        out = MaxFilter().Execute(out)
        out = sitk.GetImageFromArray(out)
        return out
    elif filter == 'LevelSet':
        np_img = sitk.GetArrayFromImage(sitk_img)
        out = LevelSet().Execute(np_img)
        out = sitk.GetImageFromArray(out)
        return out

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path', type=str, default='./features/')
    parser.add_argument('--filter', type=str, default='LevelSet',
     help=['LoG', 'Sigmoid', 'LaplacianSharp', 'BlackTop', 'WhiteTop', 'Median', 
     'Mean', 'RecGauss', 'DiscGauss', 'Gauss', 'Max', 'GaussToMax', 'LevelSet'])

    args = parser.parse_args()
    return args

def main(args):

    with open(CASE_LIST_PATH, 'r') as f:
        case_list = [row.strip() for row in f]
    
    d_lis = []
    l_lis = []
    for case in case_list:
        d_lis.append(os.path.join(DATA_DIR, 'Image', case + '.mhd'))
        l_lis.append(os.path.join(DATA_DIR, 'Label', case + '.mhd'))

    itk_im_list, itk_lb_list = load_sitk_dataset(d_lis, l_lis)

    if args.filter == 'LevelSet':
        for i, lb in enumerate(itk_lb_list):
            ed_im = filters(lb, args.filter)
            write_mhd_and_raw(ed_im, os.path.join(args.save_path, args.filter, case_list[i] + '.mhd'))
    else:
        for i, im in enumerate(itk_im_list):
            ed_im = filters(im, args.filter)
            write_mhd_and_raw(ed_im, os.path.join(args.save_path, args.filter, case_list[i] + '.mhd'))


if __name__ == '__main__':
    args = arg_parser()
    main(args)
    


