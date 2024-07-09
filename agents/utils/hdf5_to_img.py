import h5py
from io import BytesIO
from PIL import Image
import numpy as np
import os
import cv2
import torch


def read_img_from_hdf5(path, start, end, cam_resizes, device):
    f = h5py.File(os.path.join(path, 'imgs'), "r")
    cams = []
    for i, cam in enumerate(list(f.keys())):
        arr = np.array(f[cam])[start:end]
        imgs = []
        for img in arr:
            byte = BytesIO(arr)
            img = Image.open(byte)
            nparr = np.array(img)
            processed = preprocess_img_for_training(img, resize=cam_resizes[i], device=device)
            imgs.append(nparr)
        cams.append(imgs)
    f.close()
    return cams

def preprocess_img_for_training(img, resize= (256, 256), device='cuda'):

    if not img.shape == resize:
        img = cv2.resize(img, resize)

    img = img.transpose((2, 0, 1)) / 255.0

    img = torch.from_numpy(img).to(device).float().unsqueeze(0)
    
    return img


if __name__ == '__main__':
    l = [(0,1),(2,3)]
    print(l[0][1])