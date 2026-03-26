import torch
# print(torch.version.cuda)
# print(torch.cuda.get_device_name(0))

print(torch.cuda.is_available())  # should be False
print(torch.device("cpu"))