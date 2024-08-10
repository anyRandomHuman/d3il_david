import torch
import psutil
import os

# path = "/media/alr_admin/ECB69036B69002EE/Data_less_obs_space_hdf5/insertion/2024_07_02-16_13_56/imgs.hdf5"
# mem = os.path.getsize(path)
# print(mem)
t = torch.zeros((20, 20))
print(t.reshape((40,10)).shape)
