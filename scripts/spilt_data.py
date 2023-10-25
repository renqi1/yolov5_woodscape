"""
spilt data into train set and val set
"""

import os
import shutil
import random
from tqdm import tqdm

random.seed(0)


def split_data(file_path, label_path, seg_label_path, new_file_path, total, train_rate, all=False, seg_label=False):
    """
        file_path:  contain rgb images.
        label_path: contain yolo labels.
        seg_label_path: contain segmentation labels(png).
        new_file_path: new data path for training
        total: the number of images to get
        all: whether to use all data
        seg_label: whether to use segmentation labels
    """
    labels = sorted(os.listdir(label_path))
    images = [label[:-4]+'.png' for label in labels]
    data = sorted(list(zip(images, labels)))
    if all:
        total = len(images)
    random.shuffle(data)
    image, label = zip(*data)
    train_images = sorted(image[0:int(train_rate * total)])
    val_images = image[int(train_rate * total):total]
    train_labels = sorted(label[0:int(train_rate * total)])
    val_labels = label[int(train_rate * total):total]

    new_path_ti = new_file_path + '/' + 'images' + '/' + 'train'
    new_path_tl = new_file_path + '/' + 'labels' + '/' + 'train'
    new_path_vi = new_file_path + '/' + 'images' + '/' + 'val'
    new_path_vl = new_file_path + '/' + 'labels' + '/' + 'val'

    if not os.path.exists(new_path_ti):
        os.makedirs(new_path_ti)
    if not os.path.exists(new_path_tl):
        os.makedirs(new_path_tl)
    if not os.path.exists(new_path_vi):
        os.makedirs(new_path_vi)
    if not os.path.exists(new_path_vl):
        os.makedirs(new_path_vl)

    for image in tqdm(train_images):
        old_path = file_path + '/' + image
        new_path = new_path_ti + '/' + image
        shutil.copy(old_path, new_path)

    for label in tqdm(train_labels):
        old_path = label_path + '/' + label
        new_path = new_path_tl + '/' + label
        shutil.copy(old_path, new_path)

    for image in tqdm(val_images):
        old_path = file_path + '/' + image
        new_path = new_path_vi + '/' + image
        shutil.copy(old_path, new_path)

    for label in tqdm(val_labels):
        old_path = label_path + '/' + label
        new_path = new_path_vl + '/' + label
        shutil.copy(old_path, new_path)

    if seg_label:
        new_path_tl_seg = new_file_path + '/' + 'seglabels' + '/' + 'train'
        new_path_vl_seg = new_file_path + '/' + 'seglabels' + '/' + 'val'
        if not os.path.exists(new_path_tl_seg):
            os.makedirs(new_path_tl_seg)
        if not os.path.exists(new_path_vl_seg):
            os.makedirs(new_path_vl_seg)
        # image and seglabel have the same name
        for seglabel in tqdm(train_images):
            old_path = seg_label_path + '/' + seglabel
            new_path = new_path_tl_seg + '/' + seglabel
            shutil.copy(old_path, new_path)
        for seglabel in tqdm(val_images):
            old_path = seg_label_path + '/' + seglabel
            new_path = new_path_vl_seg + '/' + seglabel
            shutil.copy(old_path, new_path)


if __name__ == '__main__':
    file_path = "/home/gzz/机械硬盘/sda3/woodscape/rgb_images/"
    label_path = '/home/gzz/机械硬盘/sda3/woodscape/yolo_rotate/'
    seg_label_path = '/home/gzz/机械硬盘/sda3/woodscape/semantic_annotations/gtLabels'
    new_file_path = "/home/gzz/机械硬盘/sda3/woodscape/wood_rotate/"

    split_data(file_path, label_path, seg_label_path, new_file_path, total=2000, train_rate=0.8, all=False, seg_label=False)
