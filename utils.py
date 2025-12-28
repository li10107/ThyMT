# 对分类结果不准确的样本进行可视化
import os

import pandas as pd
from matplotlib import pyplot as plt
import torch
import numpy as np
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
import copy


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=30, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_auc, model): # __call__函数，类实例可以被用作函数，当调用类实例时，会调用这个函数。

        score = val_auc  # val_loss一般为正，score为负
        # score = -val_loss

        if self.best_score is None:   # 最开始时，best_score为None，先初始化self.best_score为-val_loss，并保存模型
            self.best_score = score
            self.save_checkpoint(val_auc, model)
        elif score < self.best_score + self.delta: # 如果score< best_score，即val_loss>last_loss，表示此时loss在上升
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # 此时val_loss在下降
            self.best_score = score
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation auc increased ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_auc_max = val_auc


def sparse_mean_std(txt_path):
    fid = open(txt_path, 'r')
    contents = fid.readlines()
    fid.close()
    means = contents[0].split()
    means_list_val = [float(x) for x in means[1::2]]
    stds = contents[1].split()
    stds_list_val = [float(x) for x in stds[1::2]]
    return means_list_val, stds_list_val


def save_feature_map(data_us, mask_us, x1, x2, x3, x4, NB_flag, save_path, us_name):
    n = x1.size(0)
    for kk in range(n):
        data_us_i, mask_us_i, x_4_i, x_3_i, x_2_i, x_1_i = data_us[kk, :, :, :], mask_us[kk, :, :], x4[kk, :, :, :], x3[kk, :, :, :], x2[kk, :, :, :], x1[kk, :, :, :]
        x_4_i, x_3_i, x_2_i, x_1_i = torch.mean(x_4_i, dim=0), torch.mean(x_3_i, dim=0), torch.mean(x_2_i, dim=0), torch.mean(x_1_i, dim=0)
        plt.subplot(2, 3, 1)
        plt.imshow(data_us_i.permute(1, 2, 0).detach().cpu().numpy())
        plt.title(NB_flag + '_us')
        plt.subplot(2, 3, 2)
        # plt.imshow(data_us_i.permute(1, 2, 0).detach().cpu().numpy(), alpha=0.9)
        plt.imshow((mask_us_i * 255).detach().cpu().numpy().astype(np.uint8), alpha=0.1)
        plt.title(NB_flag + '_us_mask')
        plt.subplot(2, 3, 3)
        plt.imshow(x_4_i.detach().cpu().numpy())
        plt.title(NB_flag + '_4')
        plt.subplot(2, 3, 4)
        plt.imshow(x_3_i.detach().cpu().numpy())
        plt.title(NB_flag + '_3')
        plt.subplot(2, 3, 5)
        plt.imshow(x_2_i.detach().cpu().numpy())
        plt.title(NB_flag + '_2')
        plt.subplot(2, 3, 6)
        plt.imshow(x_1_i.detach().cpu().numpy())
        plt.title(NB_flag + '_1')
        os.makedirs(os.path.join(save_path, 'feature map'), exist_ok=True)
        plt.savefig(os.path.join(save_path, 'feature map', us_name[kk].replace('.png', '_' + NB_flag + '.png') ))
        # plt.show()
        plt.close()


def save_anatomy_map(data_us, mask_us, x1, x2, x3, x4, NB_flag, save_path, us_name):
    n = x1.size(0)
    for kk in range(n):
        data_us_i, mask_us_i, x_4_i, x_3_i, x_2_i, x_1_i = data_us[kk, :, :, :], mask_us[kk, :, :], x4[kk, :, :, :], x3[kk, :, :, :], x2[kk, :, :, :], x1[kk, :, :, :]
        x_4_i, x_3_i, x_2_i, x_1_i = x_4_i.squeeze(), x_3_i.squeeze(), x_2_i.squeeze(), x_1_i.squeeze()
        plt.subplot(2, 3, 1)
        plt.imshow(data_us_i.permute(1, 2, 0).detach().cpu().numpy())
        plt.title(NB_flag + '_us')
        plt.subplot(2, 3, 2)
        # plt.imshow(data_us_i.permute(1, 2, 0).detach().cpu().numpy(), alpha=0.9)
        plt.imshow((mask_us_i * 255).detach().cpu().numpy().astype(np.uint8), alpha=0.1)
        plt.title(NB_flag + '_us_mask')
        plt.subplot(2, 3, 3)
        plt.imshow(x_4_i.detach().cpu().numpy())
        plt.title(NB_flag + '_4')
        plt.subplot(2, 3, 4)
        plt.imshow(x_3_i.detach().cpu().numpy())
        plt.title(NB_flag + '_3')
        plt.subplot(2, 3, 5)
        plt.imshow(x_2_i.detach().cpu().numpy())
        plt.title(NB_flag + '_2')
        plt.subplot(2, 3, 6)
        plt.imshow(x_1_i.detach().cpu().numpy())
        plt.title(NB_flag + '_1')
        os.makedirs(os.path.join(save_path, 'feature map'), exist_ok=True)
        plt.savefig(os.path.join(save_path, 'feature map', us_name[kk].replace('.png', '_' + NB_flag + '.png') ))
        # plt.show()
        plt.close()


def de_transform(img_transformed, mode):
    h, w = img_transformed.shape[2], img_transformed.shape[3]
    if mode == 'us':
        means = [76.625469, 79.033164, 83.808374]
        stds = [47.769530, 48.995110, 51.745822]
        us_mat = np.zeros([h, w, 0])
        for i in range(2):
            channel_detransformed = img_transformed[i, :, :].numpy() * stds[i] + means[i]
            us_mat = np.concatenate([us_mat, channel_detransformed.expand_dim(2)], axis=2)
        return us_mat.astype(np.uint8)
    elif mode == 'ct':
        mean_ct = 118.351912
        std_ct = 82.922086
        ct_mat = np.zeros([h, w, 0])
        for i in range(12):
            channel_detransformed = img_transformed[i, :, :].numpy() * std_ct + mean_ct
            ct_mat = np.concatenate([ct_mat, channel_detransformed.expand_dim(2)], axis=2)
        return ct_mat.astype(np.uint8)


def vis_wrong_results(filenames, img, nodule, edge, label, result, save_path):
    B = label.shape[0]
    img = img.detach().cpu()
    nodule = nodule.detach().cpu()
    edge = edge.detach().cpu()

    label = label.detach().cpu().numpy()
    result = result.detach().cpu().numpy()
    for i in range(B):

        plt.figure()
        plt.subplot(1, 3, 1)
        img_i = de_transform(img[i, :, :, :], mode='img')
        plt.imshow(img_i.numpy())
        plt.axis('off')
        plt.title('whole')

        plt.subplot(1, 3, 2)
        nodule_i = de_transform(nodule[i, :, :, :], mode='nodule')
        plt.imshow(nodule_i.numpy())
        plt.axis('off')
        plt.title('nodule')

        plt.subplot(1, 3, 3)
        edge_i = de_transform(edge[i, :, :, :], mode='edge')
        plt.imshow(edge_i.numpy())
        plt.axis('off')
        plt.title('edge')
        if label[i] != result[i]:
            plt.savefig(os.path.join(save_path, 'wrong samples', filenames[i].split('.')[0] + '_label' + str(int(label[i])) + '.jpg'))
        else:
            plt.savefig(os.path.join(save_path, 'correct samples',
                                     filenames[i].split('.')[0] + '_label' + str(int(label[i])) + '.jpg'))
        plt.close()


def gimme_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def freeze_bn(module):
    class_name = module.__class__.__name__
    if class_name.find('BatchNorm') != -1:
        module.eval()


def load_weight(model_new, model_pretrain):
    net_state_dict_keys_new = model_new.state_dict().keys()  # 自己的网络
    model_state_dict_pretrain = model_pretrain.state_dict()
    state_dict_to_load = OrderedDict()  # 有序字典
    for k, v in model_state_dict_pretrain.items():
        if k in net_state_dict_keys_new:
            if 'fc' in k:
                continue
            state_dict_to_load[k] = v
    model_new.load_state_dict(state_dict_to_load, strict=False)
    return model_new


def test():
    data_root = '/home/luye/lgj/project/202207USCTThyroidLym/干净数据集/交叉验证多模态数据集-CT数量对齐'
    for j in range(5):
        training_path = os.path.join(data_root, 'cross_validation_set' + str(j + 1), 'CT/training/cropped images')
        val_path = os.path.join(data_root, 'cross_validation_set' + str(j + 1), 'CT/validation/cropped images')
        training_files = os.listdir(training_path)
        training_pids = [x.replace('_', '-').split('-')[1] for x in training_files]
        val_files = os.listdir(val_path)
        val_pids = [x.replace('_', '-').split('-')[1] for x in val_files]
        union = set(training_pids).intersection(set(val_pids))
        print(j + 1)
        print(len(training_pids))
        print(len(val_pids))
        print(union)

def mean_std(list_data, interval):
    d = pd.Series(list_data)
    mean = list(d.rolling(interval).mean())
    std = list(d.rolling(interval).std())
    first_mean = mean[interval-1]
    first_std = std[interval-1]
    for i in range(int(interval/2)):
        mean.pop(0)
        mean[1] = first_mean
        mean.append(mean[-1])
    for i in range(int(interval/2)):
        std.pop(0)
        std[1] = first_std
        std.append(std[-1])
    std_up = np.array(mean) + np.array(std)
    std_down = np.array(mean) - np.array(std)
    return mean, std_up, std_down


def generate_sampler_weight(training_nods, label_path):
    sampler_weight_center, sampler_weight_lateral, sampler_weight_bm = [], [], []
    label = pd.read_excel(label_path, index_col=0, sheet_name=0, usecols='A:E', dtype={'A': int, 'B': bool, 'C': bool, 'D': bool, 'E': bool})
    for nod in training_nods:
        nod_id = nod.lstrip('0')
        lateral_label = label.loc[nod_id, '侧颈部淋巴结转移（转移=True，未转移=False）'].astype(bool)
        center_label = label.loc[nod_id, '中央区淋巴结转移（转移=True，未转移=False）'].astype(bool)
        bm_label = label.loc[nod_id, '病理（良=False 恶=True）'].astype(bool)
        if center_label:
            sampler_weight_center.append(0.6582397003745318)
        else:
            sampler_weight_center.append(0.34176029962546817)
        if lateral_label:
            sampler_weight_lateral.append(0.849250936329588)
        else:
            sampler_weight_lateral.append(0.150749063670412)
        if bm_label:
            sampler_weight_bm.append(0.4250936329588015)
        else:
            sampler_weight_bm.append(0.5749063670411985)
    return sampler_weight_center, sampler_weight_lateral, sampler_weight_bm


def plot_multi_results(results_train, results_val, save_path, threshold, position):
    os.makedirs(os.path.join(save_path, threshold), exist_ok=True)
    names = ['accurency', 'auc', 'sensitivity', 'specificity', 'precision', 'f1_score']
    epoch_list = results_train.iloc[:, 0]
    for i in range(1, 7):
        plt.figure()
        ptla = plt.plot(epoch_list, results_train.iloc[:, i], label=position + ' training ' + names[i-1])
        pvla = plt.plot(epoch_list, results_val.iloc[:, i], label=position + 'validation ' + names[i-1])
        plt.ylabel(threshold)
        plt.xlabel('epochs')
        plt.title(position + '-' + names[i-1] + '-' + threshold)
        l2 = plt.legend()
        plt.savefig(
            os.path.join(save_path, threshold, position + threshold + ' curve-{:.1f}.jpg'.format(i + 1 / 10)))
        plt.savefig(
            os.path.join(save_path, threshold, position + threshold + ' curve-{:.1f}.svg'.format(i + 1 / 10)),
            format='svg', transparent=True)
        plt.close()


def calculate_metric_threshold(label_list, logits_list_ori, auc): # pred：网络经过sigmoid的结果
    result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8, result_9 = [], [], [], [], [], [], [], [], []
    results = [result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8, result_9]
    for t in range(1, 10):
        thr = t / 10
        logits_list = copy.deepcopy(logits_list_ori)
        logits_list[np.where(logits_list > thr)]=1
        logits_list[np.where(logits_list < 1)]=0
        confusion = confusion_matrix(label_list, logits_list)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        acc = (TP+TN)/float(TP+TN+FP+FN)
        sen = TP / float(TP+FN)
        spe = TN / float(TN+FP)
        pre = TP / float(TP+FP)
        rec = TP / float(TP+FN)
        F1 = 2*pre*rec/(pre+rec)
        results[t - 1].extend([acc, auc, sen, spe, pre, F1])
    return results


def calculate_metric(gt, pred): # pred：网络经过sigmoid的结果
    pred[np.where(pred>0.5)]=1
    pred[np.where(pred<1)]=0
    confusion = confusion_matrix(gt,pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    acc = (TP+TN)/float(TP+TN+FP+FN)
    sen = TP / float(TP+FN)
    spe = TN / float(TN+FP)
    pre = TP / float(TP+FP)
    rec = TP / float(TP+FN)
    F1 = 2*pre*rec/(pre+rec)
    return acc, sen, spe, pre, F1


def calculate_metric_merge(gt, pred): # pred：网络经过sigmoid的结果
    confusion = confusion_matrix(gt,pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    acc = (TP+TN)/float(TP+TN+FP+FN)
    sen = TP / float(TP+FN)
    spe = TN / float(TN+FP)
    pre = TP / float(TP+FP)
    rec = TP / float(TP+FN)
    F1 = 2*pre*rec/(pre+rec)
    return acc, sen, spe, pre, F1

def cal_weight():
    path = '/home/luye/lgj/project/202207USCTThyroidLym/干净数据集/交叉验证多模态数据集-CT数量对齐/'
    weight_list = []
    for i in range(1, 6):
        weight_path = os.path.join(path + 'cross_validation_set' + str(i), 'US', 'bm_weight.npy')
        weight = np.load(weight_path)
        ratio = np.max(weight) / np.min(weight)
        weight_list.append(ratio)
    print(weight_list)








