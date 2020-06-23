import torch
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
from scipy import ndimage

class MaxFilter:

    def __init__(self, kernel_size = 5):
        self.kernel_size = kernel_size
        self.padding = kernel_size - 2
    
    def Execute(self, x):
        height, weight = x.shape
        out = np.zeros(x.shape)

        x = np.pad(x, [self.padding, self.padding], 'constant')

        for h in range(height):
            for w in range(weight):
                out[h, w] = x[h:h+self.kernel_size, w:w+self.kernel_size].max()
        
        return out


class GaussianFilter:

    def __init__(self, kernel_size = 5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kernel_size = kernel_size
        self.kernel = self.get_kernel()
    
    def get_kernel(self):

        sigma = (self.kernel_size-1)/2

        x = y = np.arange(0, self.kernel_size) - sigma

        X, Y = np.meshgrid(x,y) 

        kernel = self.norm2d(X, Y, sigma)

        return kernel

    def norm2d(self, x, y, sigma):
        Z = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        return Z.astype(np.float32)

    def Execute(self, x):
        x = torch.from_numpy(x.astype(np.float32)).clone().to(self.device)
        if len(x.shape) < 3: 
            x = x.unsqueeze(0).unsqueeze(1)

        weight = torch.from_numpy(self.kernel).clone().to(self.device)
        weight = weight.unsqueeze(0).unsqueeze(1)

        out = F.conv2d(x, weight, padding=2)

        out = out.squeeze()
        out = out.cpu().detach().numpy().copy()

        return out

class LevelSet:

    def __init__(self):
        pass

    
    def Execute(self, f):

        # Prepare the embedding function.

        # f = f > 0.5 # for nomal mode
        f = f < 0.5 # for knn

        # Signed distance transform
        dist_func = ndimage.distance_transform_edt
        distance = np.where(f, -dist_func(f) + 0.5, (dist_func(1-f) - 0.5))

        return distance


