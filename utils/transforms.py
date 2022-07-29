"""这个类实现各种预处理的变换，自己编写的主要目的是为了保证图片和标注能够处理一致"""

import numpy as np
import cv2
from .utilities import randint
# from utils.data import Transforms
from torchvision import transforms


class Transforms:
    def __init__(self, pair_functions=None, target_shape=(640, 360), convert2gray=0, rotate_range=[0,1]):
        if pair_functions is None:
            pair_functions = [
                Flip(),
                Convert2Gray(convert2gray),
                RandomRescale(),
                RandomPadding(),
                RandomShift(),
                RandomRotate(rotate_range),
                Resize(target_shape),
                transforms.ToTensor()
                # transforms.Normalize()
            ]
        self.pair_functions = pair_functions

    def __call__(self, input_pair):
        for function in self.pair_functions:
            if not isinstance(function, transforms.ToTensor):
                input_pair = function(input_pair)
                # print(function, input_pair[0].shape)
            else:
                img, mask = input_pair
                img = function(img)
                mask = function(mask)
                input_pair = [img, mask]
        return input_pair


def random_crop_byScale(input_pair, scale_ratio):
    img, mask = input_pair
    img_h, img_w, channel = img.shape
    aim_h = img_h * scale_ratio
    aim_w = img_w * scale_ratio
    aim_h, aim_w = map(round, [aim_h, aim_w])

    h_redundant = img_h - aim_h
    w_redundant = img_w - aim_w

    h_begin = randint(0, h_redundant)
    w_begin = randint(0, w_redundant)
    h_end = h_begin + aim_h
    w_end = w_begin + aim_w

    img_new = img[h_begin:h_end, w_begin: w_end]
    mask_new = mask[h_begin:h_end, w_begin:w_end]
    return img_new, mask_new


def random_padding_byScale(input_pair, scale_ratio):
    img, mask = input_pair
    img_h, img_w, channel = img.shape
    aim_h = img_h * scale_ratio
    aim_w = img_w * scale_ratio
    aim_h, aim_w = map(round, [aim_h, aim_w])

    h_add = aim_h - img_h
    w_add = aim_w - img_w

    img_new = np.zeros((aim_h, aim_w, channel), dtype=img.dtype)
    mask_new = np.zeros((aim_h, aim_w), dtype=mask.dtype)

    h_begin = randint(0, h_add)
    w_begin = randint(0, w_add)
    h_end = h_begin + img_h
    w_end = w_begin + img_w
    img_new[h_begin:h_end, w_begin:w_end] = img
    mask_new[h_begin:h_end, w_begin:w_end] = mask

    return img_new, mask_new


class RandomCrop:  # 保证长宽比的Crop
    def __init__(self, ratio=0.1):
        self.ratio = ratio

    def __call__(self, input_pair):
        scale_ratio = np.random.uniform(1 - self.ratio, 1)
        return random_crop_byScale(input_pair, scale_ratio)


class RandomPadding:
    def __init__(self, padding_limit_h=0.2, padding_limit_w=0.2, exclude_side=0):
        """exclude_side: 保持不变，不使其发生变化的边，0=top, 1=bottom, 2=left, 3=right"""
        self.padding_limit_h = padding_limit_h
        self.padding_limit_w = padding_limit_w
        self.exclude_side = exclude_side

    def random_padding(self, input_pair, padding_limit_h=0.2, padding_limit_w=0.2, exclude_side=0):
        padding_limit_h = padding_limit_h * input_pair[0].shape[0]
        padding_limit_w = padding_limit_w * input_pair[0].shape[1]
        padding_top = randint(0, padding_limit_h // 2)
        padding_bottom = randint(0, padding_limit_h // 2)
        padding_left = randint(0, padding_limit_w // 2)
        padding_right = randint(0, padding_limit_w // 2)
        padding_params = [padding_top, padding_bottom, padding_left, padding_right]
        padding_params[exclude_side] = 0
        padding_top, padding_bottom, padding_left, padding_right = padding_params
        img, mask = input_pair
        img = np.pad(img, ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)))
        mask = np.pad(mask, ((padding_top, padding_bottom), (padding_left, padding_right)))

        return img, mask

    def __call__(self, input_pair):
        return self.random_padding(input_pair, self.padding_limit_h, self.padding_limit_w, self.exclude_side)


class Flip:
    def __init__(self, p=0.5):  # 由于虚拟背景只需要考虑人脑袋在上的情况，所以只涉及左右镜像
        self.p = p

    def __call__(self, input_pair):
        temp = float(np.random.rand(1))
        flip_flag = temp < self.p
        if flip_flag:
            flip_pair = [np.flip(i, axis=1) for i in input_pair]
        else:
            flip_pair = input_pair
        return flip_pair


class RandomRescale:
    def __init__(self, crop_limit=0.1, padding_limit=0.3):  # 以后可以考虑这里的rescale考虑人的bbox必须全部在内，保证脑袋完整
        # """如果希望不做Rescale，但使用这个方法里的平移，可以令ratio=0"""
        self.crop_limit = crop_limit
        self.padding_limit = padding_limit

    def __call__(self, input_pair):
        scale_ratio = np.random.uniform(1 - self.crop_limit, 1 + self.padding_limit)
        if scale_ratio < 1:
            data_pair = random_crop_byScale(input_pair, scale_ratio)
        elif scale_ratio == 1:
            data_pair = input_pair
        else:
            data_pair = random_padding_byScale(input_pair, scale_ratio)
        return data_pair


class RandomShift:
    def __init__(self, ratio=0.15, exclude_side=0):
        """为了保证多样性，如果是会造成应保留的边移出边界，则不发生平移，只是返回原图，这样可以保留原特征，也是需要的"""
        self.ratio = ratio
        self.exclude_side = exclude_side

    def __call__(self, input_pair):
        img, mask = input_pair
        img_h, img_w, channel = img.shape
        shift_limit_h = img_h * self.ratio
        shift_limit_w = img_w * self.ratio
        h_shift = randint(-shift_limit_h, shift_limit_h)
        w_shift = randint(-shift_limit_w, shift_limit_w)

        if self.exclude_side == 0 and h_shift < 0:
            return img, mask
        elif self.exclude_side == 1 and h_shift > 0:
            return img, mask
        elif self.exclude_side == 2 and w_shift < 0:
            return img, mask
        elif self.exclude_side == 3 and w_shift > 0:
            return img, mask
        else:
            img_new = np.pad(img, ((abs(h_shift), abs(h_shift)), (abs(w_shift), abs(w_shift)), (0, 0)))
            mask_new = np.pad(mask, ((abs(h_shift), abs(h_shift)), (abs(w_shift), abs(w_shift))))
            if h_shift == 0:
                pass
            elif h_shift < 0:
                img_new = img_new[2 * abs(h_shift):]
                mask_new = mask_new[2 * abs(h_shift):]
            else:
                img_new = img_new[:-2 * h_shift]  # 需要特殊考虑h_shift==0的情况，也就是等下加的if h_shift==0
                mask_new = mask_new[:-2 * h_shift]

            if w_shift == 0:
                pass
            elif w_shift < 0:
                img_new = img_new[:, 2 * abs(w_shift):]
                mask_new = mask_new[:, 2 * abs(w_shift):]
            else:
                img_new = img_new[:, :-2 * w_shift]
                mask_new = mask_new[:, :-2 * w_shift]
            if img_new.shape[0] == 0 or img_new.shape[1] == 0:
                print('error-----------------------------------------------------')
                import time
                time.sleep(200)
            return img_new, mask_new


class Resize:
    def __init__(self, shape=(640, 360)):
        self.shape = shape

    def __call__(self, input_pair):
        output_pair = [cv2.resize(i, self.shape) for i in input_pair]
        return output_pair


class Convert2Gray:
    def __init__(self, p=0.3):  # 由于虚拟背景只需要考虑人脑袋在上的情况，所以只涉及左右镜像
        self.p = p

    def __call__(self, input_pair):
        temp = float(np.random.rand(1))
        gray_flag = temp < self.p
        img, mask = input_pair
        if gray_flag:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            gray_pair = [img, mask]
        else:
            gray_pair = input_pair
        return gray_pair


class RandomRotate:
    def __init__(self, rotate_range=[-45, 45]):
        self.rotate_range = rotate_range

    def __call__(self, input_pair):
        angle = np.random.randint(*self.rotate_range)
        img, mask = input_pair
        h, w, c = img.shape
        center = (w / 2, h / 2)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
        rotated_image = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(w, h))
        rotated_mask = cv2.warpAffine(src=mask, M=rotate_matrix, dsize=(w, h))
        rotate_pair = [rotated_image, rotated_mask]
        return rotate_pair
