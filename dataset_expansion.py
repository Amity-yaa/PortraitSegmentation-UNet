# 这个脚本是希望对原本应用的数据进行处理，得到扩展的数据集，并且生成一个近似统一的数据集。
import numpy as np
import cv2
import os
import glob
import random
import shutil
from multiprocessing import Pool

img_folder = 'image'
mask_folder = 'alpha'


# def file_add_prefix(filepath, prefix):
#     file_root, file_name = os.path.split(filepath)
#     file_name = prefix + '_' + file_name
#     return os.path.join(file_root, file_name)

def _virtualBG_generate_single(img_path, mask_path, backgrounds, generate_nums, aim_root):
    img_name = os.path.split(img_path)[-1].replace('.jpg', '.png')
    mask_name = os.path.split(mask_path)[-1].replace('.jpg', '.png')
    img_newName = 'origin_' + img_name
    mask_newName = 'origin_' + mask_name
    shutil.copy(img_path, os.path.join(aim_root, img_folder, img_newName))
    shutil.copy(mask_path, os.path.join(aim_root, mask_folder, mask_newName))

    for j in range(generate_nums):
        bg_nums = len(backgrounds)
        chosen_index = random.choice(range(bg_nums))
        background_path = backgrounds.pop(chosen_index)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)[..., :3]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        bg = cv2.imread(background_path, cv2.IMREAD_COLOR)[..., :3]
        alpha = mask / 255
        alpha = np.array([alpha] * 3).transpose((1, 2, 0))

        bg = cv2.resize(bg, (img.shape[1], img.shape[0]))
        img_new = img * alpha + bg * (1 - alpha)
        img_new = np.floor(img_new).astype(np.uint8)  # 向下取整一下，以防万一因为内部计算误差出现超过255小于256的数在uint8化出现取模，虽然可能性很低。

        img_newName = str(j) + '_' + img_name
        mask_newName = str(j) + '_' + mask_name
        cv2.imwrite(os.path.join(aim_root, img_folder, img_newName), img_new)
        shutil.copy(mask_path, os.path.join(aim_root, mask_folder, mask_newName))

        print(img_path, '{}/{}'.format(j, generate_nums), 'is ok.')


def virtualBG_generate(foregrounds, masks, backgrounds, generate_nums, aim_root):
    backgrounds_all = backgrounds.copy()
    pool = Pool(8)
    for i in range(len(foregrounds)):
        img_path = foregrounds[i]
        mask_path = masks[i]
        backgrounds = backgrounds_all.copy()

        pool.apply_async(_virtualBG_generate_single, args=(img_path, mask_path, backgrounds, generate_nums, aim_root))
    pool.close()
    pool.join()
    pass


if __name__ == '__main__':
    train_root = 'dataset/train'
    val_root = 'dataset/val'
    # if not os.path.exists(train_root):
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)
    # if not os.path.exists(os.path.join(train_root, img_folder)):
    os.makedirs(os.path.join(train_root, img_folder), exist_ok=True)
    os.makedirs(os.path.join(train_root, mask_folder), exist_ok=True)
    os.makedirs(os.path.join(val_root, img_folder), exist_ok=True)
    os.makedirs(os.path.join(val_root, mask_folder), exist_ok=True)

    backgrounds = glob.glob('backgrounds/*/*')

    # PPM数据集背景合成
    foregrounds_ppm_train = glob.glob('dataset/PPM-100/train/image/*')
    masks_ppm_train = [i.replace('image', 'matte') for i in foregrounds_ppm_train]
    # virtualBG_generate(foregrounds_ppm_train, masks_ppm_train, backgrounds, int(len(backgrounds) / 2), train_root)

    foregrounds_ppm_val = glob.glob('dataset/PPM-100/val/image/*')
    masks_ppm_val = [i.replace('image', 'matte') for i in foregrounds_ppm_val]
    virtualBG_generate(foregrounds_ppm_val, masks_ppm_val, backgrounds, int(len(backgrounds) / 2), val_root)

    # Maadaa数据集背景合成
    maadaa_PPM_ratio = 1/2  # / 3  # 希望maadaa合成出来的图的数量是PPM的三分之一

    maadaa_exclude_list = [  # 20220615-1459用于筛选maadaa的数据
        'laptop_a0152_indoor',
        'laptop_a0184_indoor',
        'laptop_a0185_indoor',
        'laptop_a0195_indoor',
        'laptop_a0198_indoor',
        'laptop_a0211_indoor',
        'laptop_a0217_indoor',
        'laptop_a0223_indoor',
        'laptop_a0225_indoor',
        'laptop_a0289_outdoor',
        'laptop_a0290_outdoor',
        'laptop_a0310_outdoor',
        'laptop_a0341_outdoor',
        'laptop_a0342_outdoor',
        'laptop_a0343_outdoor',
        'laptop_a0409_outdoor',
        'laptop_a0411_outdoor',
        # val
        'laptop_a0005_indoor',
        'laptop_a0269_outdoor',

        # 2022-0615 17:22补充
        'laptop_a0087_indoor',
        'laptop_a0093_indoor',
        'laptop_a0099_indoor',
        'laptop_a0102_indoor',
        'laptop_a0106_indoor',
        'laptop_a0107_indoor',
        'laptop_a0115_indoor',
        'laptop_a0116_indoor',
        'laptop_a0117_indoor',
        'laptop_a0119_indoor',
        'laptop_a0121_indoor',
        'laptop_a0122_indoor',
        'laptop_a0124_indoor',
        'laptop_a0125_indoor', # 很勉强可以要
        'laptop_a0134_indoor',
        'laptop_a0194_indoor',
        'laptop_a0196_indoor',
        'laptop_a0200_indoor',
        'laptop_a0227_indoor',
        'laptop_a0231_indoor',
        'laptop_a0293_outdoor',
        'laptop_a0295_outdoor',
        'laptop_a0296_outdoor',
        'laptop_a0311_outdoor',
        'laptop_a0312_outdoor',
        'laptop_a0313_outdoor',
        'laptop_a0321_outdoor',
        'laptop_a0327_outdoor',
        'laptop_a0341_outdoor',
        'laptop_a0348_outdoor',
        'laptop_a0349_outdoor',
        'laptop_a0362_outdoor',
        'laptop_a0365_outdoor',
        'laptop_a0368_outdoor',
        'laptop_a0370_outdoor',
        'laptop_a0371_outdoor',
        'laptop_a0404_outdoor',
        'laptop_a0419_outdoor',
        'laptop_a0063_indoor',
        'laptop_a0064_indoor',
        'laptop_a0243_outdoor',
        'laptop_a0255_outdoor',
        'laptop_a0256_outdoor',
        'laptop_a0257_outdoor',
        'laptop_a0258_outdoor',
        'laptop_a0260_outdoor',
        'laptop_a0263_outdoor',
        'laptop_a0269_outdoor',


    ]
    for subset_index, subset in enumerate(['train', 'val']):
        foregrounds_maadaa = []
        masks_maadaa = []
        for folder in ['indoor', 'outdoor']:
            folder_root = os.path.join('dataset', 'maadaa', subset, folder)
            mp4s = os.listdir(folder_root)
            mp4s = [i for i in mp4s if i not in maadaa_exclude_list]
            mp4s_root = [os.path.join(folder_root, i) for i in mp4s]
            for mp4_root in mp4s_root:
                masks = glob.glob(mp4_root + '/*.png')  # 有png格式的一定有对应的jpg格式（maadaa数据集里jpg是原图，png是标注）
                mask_path = random.choice(masks)
                img_path = mask_path.replace('.png', '.jpg')
                foregrounds_maadaa.append(img_path)
                masks_maadaa.append(mask_path)

        # if subset == 'train':
        PPM_num = len([foregrounds_ppm_train, foregrounds_ppm_val][subset_index]) * len(backgrounds)/2
        # else:
        #     PPM_num = len(foregrounds_ppm_val)*20
        maadaa_num = int(PPM_num * maadaa_PPM_ratio)
        maadaa_generate_num = maadaa_num // len(foregrounds_maadaa)
        # if subset == 'val':
        #     maadaa_generate_num = 20
        if subset == 'train':
            continue
        virtualBG_generate(foregrounds_maadaa,
                           masks_maadaa,
                           backgrounds,
                           maadaa_generate_num,
                           [train_root, val_root][subset_index])

        pass
