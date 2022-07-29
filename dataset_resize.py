import glob
import os
import cv2
import numpy as np
from multiprocessing import Pool


def _resize(img_path, mask_path, uniform_size, dataset_root, resized_root):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    W, H = uniform_size
    img_h, img_w, channel = img.shape
    if img_h / img_w > H / W:  # 说明高度多了，要补宽
        new_w = img_h * W / H
        w_add = new_w - img_w
        w_begin = int(w_add / 2)
        img_new = np.zeros((img_h, round(new_w), channel), dtype=np.uint8)
        img_new[:, w_begin:w_begin + img_w, :] = img

        mask_new = np.zeros((img_h, round(new_w)), dtype=np.uint8)
        mask_new[:, w_begin:w_begin + img_w] = mask

    elif img_h / img_w == H / W:
        img_new = img.copy()
        mask_new = mask.copy()

    else:
        new_h = img_w * H / W
        h_add = new_h - img_h
        h_begin = int(h_add / 2)
        img_new = np.zeros((round(new_h), img_w, channel), dtype=np.uint8)
        img_new[h_begin:h_begin + img_h, :, :] = img

        mask_new = np.zeros((round(new_h), img_w), dtype=np.uint8)
        mask_new[h_begin:h_begin + img_h, :] = mask

    img_new = cv2.resize(img_new, uniform_size)
    mask_new = cv2.resize(mask_new, uniform_size)
    img_newPath = img_path.replace(dataset_root, resized_root)
    mask_newPath = mask_path.replace(dataset_root, resized_root)

    cv2.imwrite(img_newPath, img_new)
    cv2.imwrite(mask_newPath, mask_new)

    print(img_newPath, 'is ok.')


if __name__ == '__main__':
    uniform_size = (640, 360)
    dataset_root = 'dataset'
    resized_root = 'dataset/resized'
    train_folder = 'train'
    val_folder = 'val'
    # if not os.path.exists(resized_root):
    os.makedirs(resized_root, exist_ok=True)
    for folder in [train_folder, val_folder]:
        for next_folder in ['image', 'alpha']:
            os.makedirs(os.path.join(resized_root, folder, next_folder), exist_ok=True)

    train_root = os.path.join(dataset_root, train_folder)
    val_root = os.path.join(dataset_root, val_folder)
    pool = Pool(6)
    for subset_root in [train_root, val_root]:
        if subset_root == train_root:
            continue
        img_paths = glob.glob(subset_root + '/image/*.png')
        mask_paths = [img_path.replace('image', 'alpha') for img_path in img_paths]
        for i in range(len(img_paths)):
            img_path = img_paths[i]
            mask_path = mask_paths[i]
            pool.apply_async(_resize, args=(img_path, mask_path, uniform_size, dataset_root, resized_root))
    pool.close()
    pool.join()
