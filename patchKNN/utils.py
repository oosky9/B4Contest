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
        data = normalizing(data)

    return data, label

def get_patch_dataset(data, label, patch_size):
    p_sz = patch_size
    
    assert data.shape[1] % p_sz == 0
    assert data.shape[2] % p_sz == 0

    data_set = []
    label_set = []
        
    for w in range(0, data.shape[2], p_sz):
        for h in range(0, data.shape[1], p_sz):
            data_patch = []
            label_patch = []
            for i in range(data.shape[0]):
                data_patch.append(data[i, h:h+p_sz, w:w+p_sz].flatten())
                label_patch.append(label[i, h:h+p_sz, w:w+p_sz].flatten())
            data_set.append(np.asarray(data_patch))
            label_set.append(np.asarray(label_patch))
    data_set = np.asarray(data_set)
    label_set = np.asarray(label_set)
    return data_set, label_set

def patch_integrator(patch_list, image_size, patch_size):
    i_sz = image_size
    p_sz = patch_size

    output = []
    for i in range(patch_list[0].shape[0]):
        idx = 0
        temp = np.zeros(image_size, dtype=np.int64)
        for w in range(0, i_sz[1], p_sz):
            for h in range(0, i_sz[0], p_sz):
                temp[h:h+p_sz, w:w+p_sz] = patch_list[idx][i, :].reshape(p_sz, p_sz)
                idx += 1
        output.append(temp)
    
    return output

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


