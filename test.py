import torch
import psutil
import os

<<<<<<< HEAD
# path = "/media/alr_admin/ECB69036B69002EE/Data_less_obs_space_hdf5/insertion/2024_07_02-16_13_56/imgs.hdf5"
# mem = os.path.getsize(path)
# print(mem)
t = torch.zeros((20, 20))
print(t.reshape((40,10)).shape)
=======
# ava_mem = psutil.virtual_memory().available
# print(ava_mem)
# # path = "/media/alr_admin/ECB69036B69002EE/Data_less_obs_space_hdf5/insertion/2024_07_02-16_13_56/imgs.hdf5"
# # mem = os.path.getsize(path)
# # print(mem)
# # t = torch.zeros((500, 500, 500))
# # print(t.__sizeof__())
# # print(psutil.virtual_memory().available - ava_mem)
# # del t
# # print(psutil.virtual_memory().available)
full_path = "/media/alr_admin/ECB69036B69002EE/Data_less_obs_space_hdf5/insertion/2024_07_02-16_13_56/imgs.hdf5"
path = "/media/alr_admin/ECB69036B69002EE/Data_less_obs_space_hdf5/insertion"
traj = "/2024_07_02-16_13_56/imgs.hdf5"

assert full_path == path + traj
>>>>>>> a721fed7ca2903b1f7bee02750b83a5ee418eac5
