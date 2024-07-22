import h5py
from io import BytesIO
from PIL import Image
import numpy as np
import os
import cv2
import torch


def read_img_from_hdf5(path, start, end, cam_resizes, device):
    f = h5py.File(os.path.join(path, "imgs.hdf5"), "r")
    cams = []
    for i, cam in enumerate(list(f.keys())):
        arr = f[cam][start:end]
        imgs = []
        for img in arr:
            nparr = cv2.imdecode(img, 1)
            processed = preprocess_img_for_training(
                nparr, resize=cam_resizes[i], device=device
            )
            imgs.append(processed)
        imgs = torch.concatenate(imgs, dim=0)
        cams.append(imgs)
    f.close()
    return cams


def preprocess_img_for_training(img, resize=(256, 256), device="cuda"):

    if not img.shape == resize:
        img = cv2.resize(img, resize)

    img = img.transpose((2, 0, 1)) / 255.0

    img = torch.from_numpy(img).to(device).float().unsqueeze(0)

    return img


if __name__ == "__main__":
    path = "/media/alr_admin/ECB69036B69002EE/Data_less_obs_space_hdf5/insertion/2024_07_04-16_02_30"

    cam0, cam1 = read_img_from_hdf5(path, 0, 1, [256, 256], "cpu")
