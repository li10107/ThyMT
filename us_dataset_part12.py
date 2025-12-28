# 代码功能：生成多模态数据集。
# 先读取超声中的图像数据，再根据超声读取CT中的数据
import random
import cv2
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from PIL import Image
from random import choice


class USCTDatasetTrain(Dataset):
    def __init__(self, nod_name, us_path, label_path, transform_us):

        self.nodule_name = nod_name
        self.img_path_us = os.path.join(us_path, 'image')
        self.mask_path_us = os.path.join(us_path, 'mask')
        self.boundary_path_us = os.path.join(us_path, 'boundary_10_pixels')
        self.transform_us = transform_us
        self.dict_us = {}
        us_nods = os.listdir(self.img_path_us)
        for j in range(len(us_nods)):
            nodule_name = us_nods[j]
            nod_files = os.listdir(os.path.join(self.img_path_us, nodule_name))
            if nodule_name in self.dict_us.keys():
                self.dict_us[nodule_name].extend(nod_files)
            else:
                self.dict_us[nodule_name] = nod_files

        self.label = pd.read_excel(label_path, index_col=0, sheet_name=0, usecols='A:E', dtype={'A': int, 'B': bool, 'C': bool, 'D': bool, 'E': bool})

    def __getitem__(self, idx):
        nodule_name = self.nodule_name[idx]
        nod_id = nodule_name.lstrip('0')
        center_label = self.label.loc[nod_id, '中央区淋巴结转移（转移=True，未转移=False）'].astype(bool)
        center_label = torch.tensor(center_label).float()
        lateral_label = self.label.loc[nod_id, '侧颈部淋巴结转移（转移=True，未转移=False）'].astype(bool)
        lateral_label = torch.tensor(lateral_label).float()
        bm_label = self.label.loc[nod_id, '病理（良=False 恶=True）'].astype(bool)
        bm_label = torch.tensor(bm_label).float()

        # 超声
        us_filename_list = sorted(self.dict_us[nodule_name])
        ind_us = np.random.randint(0, len(us_filename_list))
        us_filename = us_filename_list[ind_us]
        us_img_file_path = os.path.join(self.img_path_us, nodule_name, us_filename)
        us_img = Image.open(us_img_file_path)
        us_img_mat = np.array(us_img)
        us_mask_file_path = os.path.join(self.mask_path_us, nodule_name, us_filename)
        us_mask = Image.open(us_mask_file_path)
        us_mask_mat = np.array(us_mask)/255
        us_boundary_file_path = os.path.join(self.boundary_path_us, nodule_name, us_filename)
        us_boundary = Image.open(us_boundary_file_path)
        us_boundary_mat = np.array(us_boundary)/255

        us_mask_boundary = np.concatenate([np.expand_dims(us_mask_mat, axis=2), np.expand_dims(us_boundary_mat, axis=2)], axis=2)
        us_transformed = self.transform_us(image=us_img_mat, mask=us_mask_boundary)
        us_img_transformed = us_transformed['image']
        us_mask_boundary_transed = us_transformed['mask']
        us_mask_transed = us_mask_boundary_transed[:, :, 0]
        us_boundary_transed = us_mask_boundary_transed[:, :, 1]

        us_img_transformed = torch.from_numpy(us_img_transformed).float().permute(2, 0, 1)
        us_mask_transformed = torch.from_numpy(us_mask_transed).float()
        us_boundary_transformed = torch.from_numpy(us_boundary_transed).float()

        out_dict = {'us_img': us_img_transformed, 'us_mask': us_mask_transformed, 'us_boundary': us_boundary_transformed,
                    'center_label': center_label, 'lateral_label': lateral_label, 'benign_malignant_label': bm_label,
                    'nod_name': nodule_name, 'us_filename': us_filename}
        return out_dict

    def __len__(self):
        return len(self.nodule_name)


class USCTDatasetTest(Dataset):
    def __init__(self, pair_names, us_path, label_path, transform_us):

        us_name = [x.split('-nod-pair-')[1] for x in pair_names]
        self.us_names = us_name

        self.img_path_us = os.path.join(us_path, 'image')
        self.mask_path_us = os.path.join(us_path, 'mask')
        self.boundary_path_us = os.path.join(us_path, 'boundary_10_pixels')
        self.transform_us = transform_us

        self.label = pd.read_excel(label_path, index_col=0, sheet_name=0, usecols='A:E',
                                   dtype={'A': int, 'B': bool, 'C': bool, 'D': bool, 'E': bool})

    def __getitem__(self, idx):
        us_filename = self.us_names[idx]
        nodule_name = us_filename.split('-')[0]
        nod_id = nodule_name.lstrip('0')
        center_label = self.label.loc[nod_id, '中央区淋巴结转移（转移=True，未转移=False）'].astype(bool)
        center_label = torch.tensor(center_label).float()
        lateral_label = self.label.loc[nod_id, '侧颈部淋巴结转移（转移=True，未转移=False）'].astype(bool)
        lateral_label = torch.tensor(lateral_label).float()
        bm_label = self.label.loc[nod_id, '病理（良=False 恶=True）'].astype(bool)
        bm_label = torch.tensor(bm_label).float()

        # 超声
        us_img_file_path = os.path.join(self.img_path_us, nodule_name, us_filename)
        us_img = Image.open(us_img_file_path)
        us_img_mat = np.array(us_img)
        us_mask_file_path = os.path.join(self.mask_path_us, nodule_name, us_filename)
        us_mask = Image.open(us_mask_file_path)
        us_mask_mat = np.array(us_mask)/255
        us_boundary_file_path = os.path.join(self.boundary_path_us, nodule_name, us_filename)
        us_boundary = Image.open(us_boundary_file_path)
        us_boundary_mat = np.array(us_boundary)/255

        us_mask_boundary = np.concatenate([np.expand_dims(us_mask_mat, axis=2), np.expand_dims(us_boundary_mat, axis=2)], axis=2)
        us_transformed = self.transform_us(image=us_img_mat, mask=us_mask_boundary)
        us_img_transformed = us_transformed['image']
        us_mask_boundary_transed = us_transformed['mask']
        us_mask_transed = us_mask_boundary_transed[:, :, 0]
        us_boundary_transed = us_mask_boundary_transed[:, :, 1]

        us_img_transformed = torch.from_numpy(us_img_transformed).float().permute(2, 0, 1)
        us_mask_transformed = torch.from_numpy(us_mask_transed).float()
        us_boundary_transformed = torch.from_numpy(us_boundary_transed).float()

        out_dict = {'us_img': us_img_transformed, 'us_mask': us_mask_transformed, 'us_boundary': us_boundary_transformed,
                    'center_label': center_label, 'lateral_label': lateral_label, 'benign_malignant_label': bm_label,
                    'nod_name': nodule_name, 'us_filename': us_filename}
        return out_dict

    def __len__(self):
        return len(self.us_names)
