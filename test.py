import torch

a = torch.zeros((2,3,4))
b = torch.cat(a)
print(b.size())