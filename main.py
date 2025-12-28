# CLNet, hierachical Aggreation Module Ablation
import sys
sys.path.append("")
import os
import torch, math
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
import torch.nn.functional as F
from tqdm import tqdm
from losses import *
from metrics import *
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from utils import mean_std, calculate_metric, EarlyStopping
from dataset_TRFE import TN3KDataset, TN3KDatasetInstanceNormalization
from openpyxl import load_workbook
from USUnimodelSingletaskModel import ResUNet2dUSMaskBoundaryCrossLayerHierachicalAttentionNet
import torch.nn as nn


def save(path, model, prefix='best_acc'):
    name = path + '/' + 'train_epoch_models' + '_' + prefix + '.pth'
    torch.save(model.state_dict(), name)


def alhpa_seg(ep, start_epoch, max_epoch):
    theta = (ep - max_epoch) / (start_epoch - max_epoch) * 0.5 * math.pi
    a_seg = math.cos(theta)
    return a_seg


def plot_curves(plot_item_list, title_list, fig_name, save_path):
    plt.figure()
    epoch_list = list(range(len(plot_item_list[0])))
    color_list = ['r', 'b', 'g', 'm', 'y', 'c']
    for i in range(len(plot_item_list)):
        plot_item_i = plot_item_list[i]
        color_i = color_list[i]
        title_i = title_list[i]
        pt_f1_s = plt.plot(epoch_list, plot_item_i, color_i, label=title_i)
    plt.ylabel('value')
    plt.xlabel('epochs')
    plt.title(fig_name)
    plt.legend()
    plt.savefig(os.path.join(save_path, fig_name + ' .jpg'))
    plt.savefig(os.path.join(save_path, fig_name + ' .svg'), format='svg', transparent=True)
    plt.close()


def record_best_cpt(model, metric_record, metric_values, save_path):
    save_dict = {'Epoch': [metric_values['Epoch']], 'ACC': [metric_values['ACC']], 'AUC': [metric_values['AUC']],
                  'Sensibility': [metric_values['Sensibility']], 'Specificity': [metric_values['Specificity']],
                  'Precision': [metric_values['Precision']], 'f1_score': [metric_values['f1_score']], 'loss': [metric_values['loss']],
                  'dice_mask_us': [metric_values['dice_mask_us']], 'dice_boundary_us': [metric_values['dice_boundary_us']]}
    df_save = pd.DataFrame(save_dict)
    if metric_record == 'acc':
        print('Best acc:', metric_values['ACC'])
        save(save_path, model, prefix='best_acc')
        if not os.path.exists(os.path.join(save_path, "best_data.xlsx")):
            save_best_cpt = df_save
        else:
            df = pd.read_excel(os.path.join(save_path, "best_data.xlsx"))
            save_best_cpt = pd.concat([df, df_save], axis=0)
        save_best_cpt.to_excel(os.path.join(save_path, "best_data.xlsx"), index=False)
    elif metric_record == 'loss':
        print('Lowest loss:', metric_values['loss'])
        save(save_path, model, prefix='lowest_loss')
        if not os.path.exists(os.path.join(save_path, "lowest_loss_data.xlsx")):
            save_best_cpt = df_save
        else:
            df = pd.read_excel(os.path.join(save_path, "lowest_loss_data.xlsx"))
            save_best_cpt = pd.concat([df, df_save], axis=0)
        save_best_cpt.to_excel(os.path.join(save_path, "lowest_loss_data.xlsx"), index=False)


def ttest_model():
    # 利用验证集上效果最好的模型进行测试
    model_name = ['train_epoch_models_best_acc.pth', 'train_epoch_models_lowest_loss.pth']
    img_transform_test = A.Compose([A.Resize(transform_height, transform_width),
                                   A.Normalize(
                                       mean=(means_us_all[0] / 255, means_us_all[1] / 255,
                                             means_us_all[2] / 255),
                                       std=(stds_us_all[0] / 255, stds_us_all[1] / 255,
                                            stds_us_all[2] / 255))])

    test_set = TN3KDataset(mode='test', fold=fold, root=data_root, transform=img_transform_test)
    test_loader = DataLoader(test_set, shuffle=True, drop_last=False, batch_size=10, pin_memory=True, num_workers=8)
    model = ResUNet2dUSMaskBoundaryCrossLayerHierachicalAttentionNet()
    for model_name_i in model_name:
        model.load_state_dict(torch.load(os.path.join(save_path, model_name_i)))
        model.cuda()
        model.eval()

        name_list, result_list, label_list, logits_list = [], [], [], []
        loop_dice_boundary_us, loop_dice_mask_us = [], []
        for data_dict in tqdm(test_loader):
            data_us = data_dict['us_img']
            mask_texture_us = data_dict['us_mask']
            boundary_us = data_dict['us_boundary']
            label = data_dict['label']
            us_name = data_dict['filename']

            mask_texture_us = mask_texture_us.cuda()
            data_us, boundary_us, label = data_us.cuda(), boundary_us.cuda(), label.cuda()

            seg_mask_texture_us, seg_boundary_us, y_c = model(data_us)
            n = seg_boundary_us.size(0)

            pred_probabilities = F.softmax(y_c, dim=1)
            out_result = torch.argmax(pred_probabilities, dim=1)
            result_list.extend(out_result.detach().cpu().numpy())
            label_list.extend(label.detach().cpu().numpy())
            logits_list.extend(pred_probabilities[:, 1].detach().cpu().numpy())
            name_list.extend(us_name)
            for kk in range(n):
                loop_dice_mask_us.append(
                    dice_coef(seg_mask_texture_us.squeeze(1)[kk].detach().cpu(), mask_texture_us[kk].detach().cpu()))
                loop_dice_boundary_us.append(
                    dice_coef(seg_boundary_us.squeeze(1)[kk].detach().cpu(), boundary_us[kk].detach().cpu()))
        auc = roc_auc_score(label_list, logits_list)
        acc, sen, spe, precision, f1_s = calculate_metric(np.array(label_list), np.array(logits_list))
        dice_mask_us = np.mean(loop_dice_mask_us)
        dice_boundary_us = np.mean(loop_dice_boundary_us)
        result_val_to_save = pd.DataFrame(
            {'filenames': name_list, 'label': label_list, 'logits': logits_list, 'result': result_list})
        result_val_to_save.to_excel(os.path.join(save_path, 'test_result.xlsx'))
        print(model_name_i.rstrip('.pth'), 'ACC:', acc, 'AUC:', auc, 'Sensibility:', sen, 'Specificity:', spe,
              'Precision:', precision, 'f1_score:', f1_s, 'dice_mask_us:', dice_mask_us, 'dice_boundary_us:', dice_boundary_us)
        # save to txt
        with open(os.path.join(save_path, 'test_result' + model_name_i.rstrip('.pth') + '.txt'), 'w') as f:
            f.write(
                model_name_i.rstrip('.pth') +
                '\nACC\tAUC\tSensibility\tSpecificity\tPrecision\tf1_score\tdice_mask_us\tdice_boundary_us' +
                f'\n{acc} \t {auc} \t {sen} \t {spe} \t {precision} \t {f1_s} \t {dice_mask_us} \t {dice_boundary_us}')

def train_val():
    # torch.backends.cudnn.benchmark = True

    base_lr = 0.0001
    max_epoch = 600 # 600
    batch_size = 10  # us-40: 15, full: 5
    start_epoch = 100 # 200

    # Section: 数据集相关
    img_transform_train = A.Compose([A.Resize(transform_height, transform_width),
                                     A.Rotate(limit=(-15, 15), p=0.5, value=0),
                                     A.HorizontalFlip(p=0.5),
                                     A.RandomResizedCrop(height=transform_height, width=transform_width, scale=(0.9, 1)),
                                     A.RandomBrightnessContrast(),
                                     A.ElasticTransform(),
                                     A.Normalize(
                                            mean=(means_us_all[0] / 255, means_us_all[1] / 255, means_us_all[2] / 255),
                                            std=(stds_us_all[0] / 255, stds_us_all[1] / 255, stds_us_all[2] / 255))])

    img_transform_val = A.Compose([A.Resize(transform_height, transform_width),
                                   A.Normalize(
                                          mean=(means_us_all[0] / 255, means_us_all[1] / 255,
                                                means_us_all[2] / 255),
                                          std=(stds_us_all[0] / 255, stds_us_all[1] / 255,
                                               stds_us_all[2] / 255))])


    train_set = TN3KDataset(mode='train', fold=fold, root=data_root, transform=img_transform_train)
    val_set = TN3KDataset(mode='val', fold=fold, root=data_root, transform=img_transform_val)

    # 3. Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_set, shuffle=True, drop_last=False, batch_size=batch_size, pin_memory=True, num_workers=8)

    # Section: 模型相关
    model = ResUNet2dUSMaskBoundaryCrossLayerHierachicalAttentionNet()
    model.cuda()

    # Section：定义优化器
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, model.parameters()), lr=base_lr, weight_decay=0.00001)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, round(max_epoch * 0.5) * len(train_loader), 0)
    # 训练和验证

    class_loss_train, correct_train, auc_train, sensitivity_train, specificity_train, precision_train, f1_score_train = [], [], [], [], [], [], []
    class_loss_val, correct_val, auc_val, sensitivity_val, specificity_val, precision_val, f1_score_val = [], [], [], [], [], [], []

    boundary_loss_train_us = []
    mask_loss_train_us = []
    dice_train_us_boundary = []
    dice_train_us_mask = []

    boundary_loss_val_us = []
    mask_loss_val_us = []
    dice_val_us_boundary = []
    dice_val_us_mask = []

    best_acc = 0
    best_auc = 0
    lowest_val_loss = 100
    best_epoch = 0

    epoch_list = []
    epoch_train = 0
    lr_list = []
    mode = 'train'

    criterion_seg = nn.BCEWithLogitsLoss(reduction='none')

    # early_stopping = EarlyStopping(patience=70, verbose=True, path=os.path.join(save_path, 'checkpoint.pt'))

    for ep in range(1, max_epoch + 1):

        if mode == 'train':
            model.train()
            data_loader = train_loader
        else:
            model.eval()
            data_loader = val_loader
        loop_loss_class, correct, result, AUC, label_list, logits_list = [], [], [], [], [], []
        loop_loss_boundary_us, loop_dice_boundary_us = [], []
        loop_loss_mask_us, loop_dice_mask_us = [], []
        us_name_list = []
        for data_dict in tqdm(data_loader):
            data_us = data_dict['us_img']
            mask_texture_us = data_dict['us_mask']
            boundary_us = data_dict['us_boundary']
            label = data_dict['label']
            us_name = data_dict['filename']

            mask_texture_us = mask_texture_us.cuda()
            data_us, boundary_us, label = data_us.cuda(), boundary_us.cuda(), label.cuda()

            model.cuda()
            if mode == 'train':
                torch.set_grad_enabled(True)
            else:
                torch.set_grad_enabled(False)
            seg_mask_texture_us, seg_boundary_us, y_c = model(data_us)

            weight = torch.tensor([974, 1905]).float().cuda()
            loss_class = nn.functional.cross_entropy(y_c, label, weight=weight)
            n = seg_boundary_us.size(0)

            loss_mask_texture_us = torch.mean(criterion_seg(seg_mask_texture_us.squeeze(1), mask_texture_us)) + IOULoss(seg_mask_texture_us.squeeze(1), mask_texture_us)
            loss_boundary_us = torch.mean(criterion_seg(seg_boundary_us.squeeze(1), boundary_us)) + IOULoss(seg_boundary_us.squeeze(1), boundary_us)


            a_class = alpha
            a_us_mask_texture = 0.2
            a_us_boundary = 0.2
            loss = a_class * loss_class + a_us_mask_texture * loss_mask_texture_us + a_us_boundary * loss_boundary_us
            us_name_list.extend(us_name)

            loop_loss_class.append(a_class * loss_class.detach().cpu().numpy() / (len(data_loader) * batch_size))
            pred_probabilities = F.softmax(y_c, dim=1)
            out_result = torch.argmax(pred_probabilities, dim=1)
            out = (out_result == label.data)
            result.extend(out_result.cpu().numpy())
            correct.extend(out.cpu().numpy())
            label_list.extend(label.detach().cpu().numpy())
            logits_list.extend(pred_probabilities[:, 1].detach().cpu().numpy())

            loop_loss_mask_us.append(a_us_mask_texture * loss_mask_texture_us.cpu().data / (len(data_loader) * batch_size))
            loop_loss_boundary_us.append(a_us_boundary * loss_boundary_us.cpu().data / (len(data_loader) * batch_size))

            for kk in range(n):
                loop_dice_mask_us.append(dice_coef(seg_mask_texture_us.squeeze(1)[kk].detach().cpu(), mask_texture_us[kk].detach().cpu()))
                loop_dice_boundary_us.append(dice_coef(seg_boundary_us.squeeze(1)[kk].detach().cpu(), boundary_us[kk].detach().cpu()))

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if scheduler is not None:
                #     lr_list.append(scheduler.get_last_lr()[0])
                #     scheduler.step()

        if mode == 'train':
            mask_loss_train_us.append(sum(loop_loss_mask_us))
            boundary_loss_train_us.append(sum(loop_loss_boundary_us))
            dice_train_us_boundary.append(sum(loop_dice_boundary_us) / len(loop_dice_boundary_us))
            dice_train_us_mask.append(sum(loop_dice_mask_us) / len(loop_dice_mask_us))

            class_loss_train.append(sum(loop_loss_class))
            auc = roc_auc_score(label_list, logits_list)
            acc, sen, spe, precision, f1_s = calculate_metric(np.array(label_list), np.array(logits_list))
            correct_train.append(acc)
            auc_train.append(auc)
            sensitivity_train.append(sen)
            specificity_train.append(spe)
            precision_train.append(precision)
            f1_score_train.append(f1_s)

            epoch_list.append(epoch_train)
            epoch_train = epoch_train + 1
        else:
            boundary_loss_val_us.append(sum(loop_loss_boundary_us))
            mask_loss_val_us.append(sum(loop_loss_mask_us))
            dice_val_us_boundary.append(sum(loop_dice_boundary_us) / len(loop_dice_boundary_us))
            dice_val_us_mask.append(sum(loop_dice_mask_us) / len(loop_dice_mask_us))

            class_loss_val.append(sum(loop_loss_class))
            auc = roc_auc_score(label_list, logits_list)
            acc, sen, spe, precision, f1_s = calculate_metric(np.array(label_list), np.array(logits_list))
            correct_val.append(acc)
            auc_val.append(auc)
            sensitivity_val.append(sen)
            specificity_val.append(spe)
            precision_val.append(precision)
            f1_score_val.append(f1_s)

            metric_values = {'Epoch': ep, 'ACC': acc, 'AUC': auc, 'Sensibility': sen, 'Specificity': spe,
                             'Precision': precision, 'f1_score': f1_s, 'loss': sum(loop_loss_class),
                             'dice_mask_us': np.mean(loop_dice_mask_us), 'dice_boundary_us': np.mean(loop_dice_boundary_us)}
            if acc >= best_acc and ep > start_epoch:
                best_acc = acc
                record_best_cpt(model, metric_record='acc', metric_values=metric_values, save_path=save_path)

                result_val_to_save = pd.DataFrame(
                    {'filenames': us_name_list, 'label': label_list, 'logits': logits_list})
                result_val_to_save.to_excel(os.path.join(save_path, 'validation_result_' + str(ep) + '.xlsx'))

            if sum(loop_loss_class) < lowest_val_loss and ep > start_epoch:
                lowest_val_loss = sum(loop_loss_class)
                record_best_cpt(model, metric_record='loss', metric_values=metric_values, save_path=save_path)
                result_val_to_save = pd.DataFrame(
                    {'filenames': us_name_list, 'label': label_list, 'logits': logits_list})
                result_val_to_save.to_excel(os.path.join(save_path, 'validation_result_' + str(ep) + '.xlsx'))

            # if ep > start_epoch:
            #     early_stopping(acc, model)

            #     if early_stopping.early_stop:
            #         print("Early stopping")
            #         break

        print(mode + ': ep: {:03d}, loss: {:.6f}, loss_mask_us: {:.6f}, loss_boundary_us: {:.6f}, '
                     'Acc: {:.6%}, AUC: {:.6%}, mask_dice_us: {:.6f}, boundary_dice_us: {:.6f}'.format(ep, sum(loop_loss_class), sum(loop_loss_mask_us), sum(loop_loss_boundary_us),
                                                                                                       acc, auc, np.mean(loop_dice_mask_us), np.mean(loop_dice_boundary_us)))

        if mode == 'train':
            mode = 'test'
        else:
            mode = 'train'


    fid = open(os.path.join(save_path, 'training log.txt'), 'w')
    contents_titles_best = ['Epoch\t', 'ACC\t', 'AUC\t', 'Sensibility\t', 'Specificity\t', 'Precision\t', 'f1_score\t', 'mask\t' + 'Dice US\t', 'boundary Dice US\n']
    for i in range(9):
        fid.write(contents_titles_best[i])
    for jj in range(len(correct_val)):
        fid.write('{:.5f}'.format(epoch_list[jj]) + '\t' + '{:.5f}'.format(correct_val[jj]) + '\t' + '{:.5f}'.format(auc_val[jj]) + '\t' + '{:.5f}'.format(sensitivity_val[jj]) + '\t' +
                  '{:.5f}'.format(specificity_val[jj]) + '\t' + '{:.5f}'.format(precision_val[jj]) + '\t' + '{:.5f}'.format(f1_score_val[jj]) + '\t' +
                  '{:.5f}'.format(dice_val_us_mask[jj]) + '\t' + '{:.5f}'.format(dice_val_us_boundary[jj]) + '\n')
    fid.close()

    plot_curves([class_loss_train, boundary_loss_train_us, mask_loss_train_us], ['classification training loss', 'Boundary segmentation training loss', 'Mask segmentation training loss'],
                fig_name='training loss', save_path=save_path)
    plot_curves([class_loss_val, boundary_loss_val_us, mask_loss_val_us], ['classification validation loss', 'Boundary segmentation validation loss', 'Mask segmentation validation loss'],
                fig_name='validation loss', save_path=save_path)

    plot_curves([lr_list], ['learning rate'], fig_name='learning rate', save_path=save_path)
    plot_curves([boundary_loss_train_us, boundary_loss_val_us], ['Boundary segmentation training loss', 'Boundary segmentation validation loss'],
                fig_name='US boundary loss', save_path=save_path)
    plot_curves([dice_train_us_boundary, dice_val_us_boundary], ['Boundary segmentation training dice', 'Boundary segmentation validation dice'],
                fig_name='US boundary dice', save_path=save_path)
    plot_curves([mask_loss_train_us, mask_loss_val_us], ['Mask segmentation training loss', 'Mask segmentation validation loss'],
                fig_name='US mask loss', save_path=save_path)
    plot_curves([dice_train_us_mask, dice_val_us_mask], ['Mask segmentation training dice', 'Mask segmentation validation dice'],
                fig_name='US mask dice', save_path=save_path)

    plot_curves([class_loss_train, class_loss_val], ['classification training loss', 'classification validation loss'],
                fig_name='classification loss', save_path=save_path)
    plot_curves([correct_train, correct_val], ['training accuracy', 'validation accuracy'],
                fig_name='classification accuracy', save_path=save_path)
    plot_curves([auc_train, auc_val], ['training AUC', 'validation AUC'], fig_name='classification AUC', save_path=save_path)
    plot_curves([sensitivity_train, sensitivity_val], ['training sensitivity', 'validation sensitivity'], fig_name='classification sensitivity', save_path=save_path)
    plot_curves([specificity_train, specificity_val], ['training specificity', 'validation specificity'], fig_name='classification specificity', save_path=save_path)
    plot_curves([precision_train, precision_val], ['training precision', 'validation precision'], fig_name='classification precision', save_path=save_path)
    plot_curves([f1_score_train, f1_score_val], ['training f1_score', 'validation f1_score'], fig_name='classification f1_score', save_path=save_path)


if __name__ == '__main__':
    data_root = '/t9k/mnt/0711_USUnomdal/data/外部验证/TRFE/datasets/tn3k'
    save_path = '/t9k/mnt/0711_USUnomdal/外部验证实验/TRFE/Results/整体归一化-重复实验'
    os.makedirs(save_path, exist_ok=True)

    transform_height = 384
    transform_width = 448
    means_us_all = (65.196631, 65.196631, 65.196631)
    stds_us_all = (51.382809, 51.382809, 51.382809)

    train_val()
    ttest_model()
