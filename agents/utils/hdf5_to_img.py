import h5py
from io import BytesIO
from PIL import Image
import numpy as np
import os

Class
def read_img_from_hdf5(path):
    f = h5py.File(os.path.join(path, 'imgs'), "r")
    for cam in list(f.keys()):
        arr = np.array(f[cam])
        imgs = []
        for img in arr:
            byte = BytesIO(arr)
            img = Image.open(byte)
            nparr = np.array(img)
            imgs.append(nparr)
    f.close()
