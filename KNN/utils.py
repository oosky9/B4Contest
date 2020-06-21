import SimpleITK as sitk
import numpy as np

import os

NUM_OF_DATA = 50

def normalizing(x):
    mu = np.mean(x)
    sigma = np.std(x)
    x = (x - mu)/sigma
    return x

def load_dataset(data_list, label_list, normalize=False, flat=True):

    data = []
    label = []
    for d in data_list:
        itk_img = sitk.ReadImage(d)
        img = sitk.GetArrayFromImage(itk_img)
        if flat:
            data.append(img.flatten().astype(np.float32))
        else:
            data.append(img.astype(np.float32))

    for l in label_list:
        itk_gt = sitk.ReadImage(l)
        gt = sitk.GetArrayFromImage(itk_gt).astype(bool)
        if flat:
            label.append(gt.flatten().astype(np.int64))
        else:
            label.append(gt.astype(np.int64))

    data = np.asarray(data)
    label = np.asarray(label)

    if normalize:
        d = []
        if data.shape[0] > NUM_OF_DATA:
            for i in range(data.shape[0]//NUM_OF_DATA):
                d.append(normalizing(data[i*NUM_OF_DATA:(i+1)*NUM_OF_DATA]))
            data = np.concatenate(d, axis=0)
        else:
            data = normalizing(data)

    return data, label


def load_sitk_dataset(data_list):

    data = []
    for d in data_list:
        itk_img = sitk.ReadImage(d)
        data.append(itk_img)
    
    return data

def write_mhd_and_raw(Data, path):
    """
    This function use sitk
    Data : sitkArray
    path : Meta data path
    ex. /hogehoge.mhd
    """

    if not isinstance(Data, sitk.SimpleITK.Image):
        print('Please check your ''Data'' class')
        return False

    data_dir, file_name = os.path.split(path)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    sitk.WriteImage(Data, path, True)

    return True


