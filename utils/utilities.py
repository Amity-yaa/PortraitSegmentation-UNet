import numpy as np
import torch


def randint(low, high):  # np.random.randint在low=high的时候会报错
    if low == high:
        return low
    else:
        return np.random.randint(low, high)


class Logger:
    def __init__(self, log_path):
        self.log_file = log_path
        self.log = open(log_path, 'a')

    def __call__(self, content):
        self.log.write(content + '\n')
        print(content)

    def __del__(self):
        self.log.close()


def BGR2Tensor(img, device='cuda'):
    img = img.transpose((2, 0, 1))[None, :, :, :]
    img = torch.tensor(img / 255, dtype=torch.float32).to(device)
    return img
