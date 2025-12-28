import json
import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
import cv2
import random
import shutil


def get_bbox(mask):
    mask = cv2.imread(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    stats = stats[1][:4]
    return stats


def make_dataset(root, seed):
    imgs = []
    img_labels = {}

    # get label dict
    with open(root + '/label4trainval.csv', 'r') as f:
        lines = f.readlines()

    for idx in range(len(lines)):
        line = lines[idx]
        name, label = line.strip().split(',')
        img_labels[name] = label  # Note: img_labels, key is img_name, value is label
    # get image path
    img_names = os.listdir(root + '/trainval-image/')
    # 删除'.ipynb_checkpoints'文件
    if '.ipynb_checkpoints' in img_names:
        img_names.remove('.ipynb_checkpoints')
        shutil.rmtree(os.path.join(root + '/trainval-image/', '.ipynb_checkpoints'))
    for i in seed:  # Note: seed is the training set index
        img_name = img_names[i]
        img = os.path.join(root + '/trainval-image/', img_name)
        mask = os.path.join(root + '/trainval-mask/', img_name)
        boundary = os.path.join(root + '/trainval-boundary/', img_name)
        # if int(img_labels[img_name]) == 1:
        #     imgs.append((img, mask, img_labels[img_name]))  # Note: 这里原来将标签为1的图片添加了两次，这里注释掉了
        imgs.append((img, mask, boundary, img_labels[img_name]))  # Note: imgs, value is a tuple, (img_path, mask_path, boundary_path, label)
    return imgs


def make_validset(root, seed):
    imgs = []
    img_labels = {}

    # get label dict
    with open(root + '/label4trainval.csv', 'r') as f:
        lines = f.readlines()

    for idx in range(len(lines)):
        line = lines[idx]
        name, label = line.strip().split(',')
        img_labels[name] = label
    # get image path
    img_names = os.listdir(root + '/trainval-image/')
    if '.ipynb_checkpoints' in img_names:
        img_names.remove('.ipynb_checkpoints')
        shutil.rmtree(os.path.join(root + '/trainval-image/', '.ipynb_checkpoints'))
    for i in seed:
        img_name = img_names[i]
        img = os.path.join(root + '/trainval-image/', img_name)
        mask = os.path.join(root + '/trainval-mask/', img_name)
        boundary = os.path.join(root + '/trainval-boundary/', img_name)
        imgs.append((img, mask, boundary, img_labels[img_name]))
    return imgs


def make_testset(root):
    imgs = []
    img_labels = {}

    # get label dict
    with open(root + '/label4test.csv', 'r') as f:
        lines = f.readlines()

    for idx in range(0, len(lines)):
        line = lines[idx]
        name, label = line.strip().split(',')
        img_labels[name] = label

    # get image path
    img_names = os.listdir(root + '/test-image/')
    if '.ipynb_checkpoints' in img_names:
        img_names.remove('.ipynb_checkpoints')
        shutil.rmtree(os.path.join(root + '/trainval-image/', '.ipynb_checkpoints'))
    for img_name in img_names:
        img = os.path.join(root + '/test-image/', img_name)
        mask = os.path.join(root + '/test-mask/', img_name)
        boundary = os.path.join(root + '/test-boundary/', img_name)
        imgs.append((img, mask, boundary, img_labels[img_name]))
    return imgs


class TN3KDataset(data.Dataset):
    def __init__(self, mode='train', transform=None, root=None, return_size=False, fold=0):
        self.mode = mode
        self.transform = transform
        self.return_size = return_size

        trainvaltest = json.load(open(root + '/tn3k-trainval-fold' + str(fold) + '.json', 'r'))

        if mode == 'train':
            imgs = make_dataset(root, trainvaltest['train'])
        elif mode == 'val':
            imgs = make_validset(root, trainvaltest['val'])
        elif mode == 'test':
            imgs = make_testset(root)

        self.imgs = imgs
        self.transform = transform
        self.return_size = return_size

    def __getitem__(self, item):

        image_path, mask_path, boundary_path, label = self.imgs[item]

        assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
        assert os.path.exists(mask_path), ('{} does not exist'.format(mask_path))

        us_img = Image.open(image_path).convert('RGB')
        us_img_mat = np.array(us_img)
        us_mask = Image.open(mask_path)
        us_mask_mat = np.array(us_mask) / 255
        us_boundary = Image.open(boundary_path)
        us_boundary_mat = np.array(us_boundary) / 255

        us_mask_boundary = np.concatenate(
            [np.expand_dims(us_mask_mat, axis=2), np.expand_dims(us_boundary_mat, axis=2)], axis=2)
        us_transformed = self.transform(image=us_img_mat, mask=us_mask_boundary)
        us_img_transformed = us_transformed['image']
        us_mask_boundary_transed = us_transformed['mask']
        us_mask_transed = us_mask_boundary_transed[:, :, 0]
        us_boundary_transed = us_mask_boundary_transed[:, :, 1]

        us_img_transformed = torch.from_numpy(us_img_transformed).float().permute(2, 0, 1)
        us_mask_transformed = torch.from_numpy(us_mask_transed).float()
        us_boundary_transformed = torch.from_numpy(us_boundary_transed).float()

        filename = os.path.basename(image_path)

        out_dict = {'us_img': us_img_transformed, 'us_mask': us_mask_transformed,
                    'us_boundary': us_boundary_transformed, 'label': int(label), 'filename': filename}
        return out_dict

    def __len__(self):
        return len(self.imgs)


class TN3KDatasetInstanceNormalization(data.Dataset):
    def __init__(self, mode='train', transform=None, root=None, return_size=False, fold=0):
        self.mode = mode
        self.transform = transform
        self.return_size = return_size

        trainvaltest = json.load(open(root + '/tn3k-trainval-fold' + str(fold) + '.json', 'r'))

        if mode == 'train':
            imgs = make_dataset(root, trainvaltest['train'])
        elif mode == 'val':
            imgs = make_validset(root, trainvaltest['val'])
        elif mode == 'test':
            imgs = make_testset(root)

        self.imgs = imgs
        self.transform = transform
        self.return_size = return_size

    def __getitem__(self, item):

        image_path, mask_path, boundary_path, label = self.imgs[item]

        assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
        assert os.path.exists(mask_path), ('{} does not exist'.format(mask_path))

        us_img = Image.open(image_path).convert('RGB')
        us_img_mat = np.array(us_img)
        us_mask = Image.open(mask_path)
        us_mask_mat = np.array(us_mask) / 255
        us_boundary = Image.open(boundary_path)
        us_boundary_mat = np.array(us_boundary) / 255

        us_mask_boundary = np.concatenate(
            [np.expand_dims(us_mask_mat, axis=2), np.expand_dims(us_boundary_mat, axis=2)], axis=2)
        us_transformed = self.transform(image=us_img_mat, mask=us_mask_boundary)
        us_img_transformed = us_transformed['image']
        us_mask_boundary_transed = us_transformed['mask']
        us_mask_transed = us_mask_boundary_transed[:, :, 0]
        us_boundary_transed = us_mask_boundary_transed[:, :, 1]

        # instance normalization
        img_mean = np.mean(us_img_transformed.reshape(-1, 3), axis=0)
        img_std = np.std(us_img_transformed.reshape(-1, 3), axis=0)
        us_img_transformed = (us_img_transformed - img_mean) / img_std

        us_img_transformed = torch.from_numpy(us_img_transformed).float().permute(2, 0, 1)
        us_mask_transformed = torch.from_numpy(us_mask_transed).float()
        us_boundary_transformed = torch.from_numpy(us_boundary_transed).float()

        filename = os.path.basename(image_path)

        out_dict = {'us_img': us_img_transformed, 'us_mask': us_mask_transformed,
                    'us_boundary': us_boundary_transformed, 'label': int(label), 'filename': filename}
        return out_dict

    def __len__(self):
        return len(self.imgs)