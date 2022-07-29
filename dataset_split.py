"""
此脚本用于切分数据集，需要修改验证集的比例可以修改maadaa_split_ratio和PPM_split_ratio
"""
import os
import glob
import cv2
import shutil

if __name__ == '__main__':
    origin_root = 'dataset_origin'
    new_root = 'dataset'
    if not os.path.exists(new_root):
        os.mkdir(new_root)
        for folder_name in ['train', 'val']:
            os.mkdir(os.path.join(new_root, folder_name))

    # maadaa数据集划分
    maadaa_split_ratio = 0.2  # 切出去作为验证集的比例
    maadaa_folder = ['indoor', 'outdoor']
    maadaa_dataset = os.path.join(origin_root, 'maadaa', 'train')
    for folder_name in maadaa_folder:
        folder_root = os.path.join(maadaa_dataset, folder_name)
        mp4s = os.listdir(folder_root)
        val_num = int(maadaa_split_ratio * len(mp4s))
        val_mp4s = mp4s[:val_num]
        train_mp4s = mp4s[val_num:]
        for i, mp4s_subset in enumerate([train_mp4s, val_mp4s]):
            for mp4_name in mp4s_subset:
                files = glob.glob(os.path.join(folder_root, mp4_name, '*.jpg'))
                for file in files:
                    file_newPath = file.replace(origin_root, new_root)
                    if i == 1:
                        file_newPath = file_newPath.replace(os.path.sep + 'train' + os.path.sep,
                                                            os.path.sep + 'val' + os.path.sep)
                    file_newRoot = os.path.split(file_newPath)[0]
                    if not os.path.exists(file_newRoot):
                        os.makedirs(file_newRoot, exist_ok=True)

                    mask_file = file.replace('.jpg', '.png')
                    mask_newPath = file_newPath.replace('.jpg', '.png')
                    if os.path.exists(mask_file):  # 详见README.md的问题1，因为file是glob.glob获取的，所以一定存在，所以只需要判断一下mask是否存在。
                        shutil.copy(file, file_newPath)
                        shutil.copy(mask_file, mask_newPath)
                    print(file, 'is ok.')

    # PPM数据集划分
    PPM_split_ratio = 0.1
    PPM_dataset = os.path.join(origin_root, 'PPM-100')
    PPM_img_root = os.path.join(PPM_dataset, 'image')
    PPM_imgs = glob.glob(PPM_img_root + '/*.jpg')
    val_num = int(PPM_split_ratio * len(PPM_imgs))
    val_PPM = PPM_imgs[:val_num]
    train_PPM = PPM_imgs[val_num:]
    for i, PPM_subset in enumerate([train_PPM, val_PPM]):
        folder = ['train', 'val'][i]
        for img_path in PPM_subset:
            mask_path = img_path.replace('image', 'matte')
            img_newPath = img_path.replace(origin_root, new_root).replace('PPM-100', 'PPM-100' + os.path.sep + folder)
            mask_newPath = mask_path.replace(origin_root, new_root).replace('PPM-100', 'PPM-100' + os.path.sep + folder)
            img_root = os.path.split(img_newPath)[0]
            mask_root = os.path.split(mask_newPath)[0]
            if not os.path.exists(img_root):
                os.makedirs(img_root, exist_ok=True)
                os.makedirs(mask_root, exist_ok=True)
            shutil.copy(img_path, img_newPath)
            shutil.copy(mask_path, mask_newPath)
            print(img_path, 'is ok.')
