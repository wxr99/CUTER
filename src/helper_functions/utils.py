import sys
from torch.utils.tensorboard import SummaryWriter
from src.helper_functions.logger import setup_logger
import os 
import time
import pandas as pd
import PIL.Image as Image
import torch
import torch.nn.functional as F

def resize_pil(I, patch_size=16) :
    w, h = I.size

    new_w, new_h = int(round(w / patch_size)) * patch_size, int(round(h / patch_size)) * patch_size
    feat_w, feat_h = new_w // patch_size, new_h // patch_size

    return I.resize((new_w, new_h), resample=Image.LANCZOS), w, h, feat_w, feat_h

def build_logger(logger_dir, rank):
    logger = setup_logger(output=logger_dir + '/log',
                          distributed_rank=rank, color=False, name="MultiLabelIncremental")
    logger.info("Command: " + ' '.join(sys.argv))
    return logger


def build_writer(tensorboard_dir, args):
    writer = SummaryWriter(tensorboard_dir)
    writer.add_text('Args', str(vars(args)))

    return writer


def print_to_excel(excel_path, expe_name, dataset_name, base_classes, task_size, total_classes, params, map, cf1, of1, metrics, git_hash):
    # read excel content
    sheet_name=f"{dataset_name} {base_classes}+{task_size}"

    # Results
    incremental_stages = [(0, base_classes)] + [
                (low, low + task_size) for low in range(base_classes, total_classes, task_size)]
    columns = ['date', 'name', 'params']
    for low_range, high_range in incremental_stages:
        columns.append(f"{low_range}-{high_range}")
    columns.append('avg_mAP')
    columns.append('avg_cf1')
    columns.append('avg_of1')
    # for metric in ["precision_c", "recall_c", "f1_c", "precision_o", "recall_o", "f1_o"]:
    #     columns.append(metric)
    columns.append('git hash')

    Current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    avg_map = sum(map)/len(map)
    avg_cf1 = sum(cf1) / len(cf1)
    avg_of1 = sum(of1) / len(of1)
    new_result = pd.DataFrame([[Current_time, expe_name, params, *map, avg_map, avg_cf1, avg_of1, git_hash]], columns=columns)
    
    result = new_result
    if os.path.exists(excel_path):
        old_results = pd.read_excel(excel_path, sheet_name=None)
        if sheet_name in old_results.keys():
            old_results = old_results[sheet_name]
            result = old_results.append(new_result)
        with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
            result.to_excel(writer, f'{dataset_name} {base_classes}+{task_size}', index=False)
    
    else:
        with pd.ExcelWriter(excel_path, mode='w', engine='openpyxl') as writer:
            result.to_excel(writer, f'{dataset_name} {base_classes}+{task_size}', index=False)


def calculate_metrics(preds, targets, thre):

    prediction = preds.gt(thre).long()
    tp_c = (prediction + targets).eq(2).sum(dim=0)
    fp_c = (prediction - targets).eq(1).sum(dim=0)
    fn_c = (prediction - targets).eq(-1).sum(dim=0)
    tn_c = (prediction + targets).eq(0).sum(dim=0)
    count = targets.size(0)

    precision_c = [float(tp_c[i].float() / (tp_c[i] + fp_c[i]).float()) * 100.0 if tp_c[i] > 0 else 0.0 for i in range(len(tp_c))]
    recall_c = [float(tp_c[i].float() / (tp_c[i] + fn_c[i]).float()) * 100.0 if tp_c[i] > 0 else 0.0 for i in range(len(tp_c))]
    f1_c = [2 * precision_c[i] * recall_c[i] / (precision_c[i] + recall_c[i]) if tp_c[i] > 0 else 0.0 for i in range(len(tp_c))]

    mean_p_c = sum(precision_c) / len(precision_c)
    mean_r_c = sum(recall_c) / len(recall_c)
    mean_f_c = sum(f1_c) / len(f1_c)

    precision_o = tp_c.sum().float() / (tp_c + fp_c).sum().float() * 100.0
    recall_o = tp_c.sum().float() / (tp_c + fn_c).sum().float() * 100.0
    f1_o = 2 * precision_o * recall_o / (precision_o + recall_o)

    recall_o = tp_c.sum().float() / (tp_c + fn_c).sum().float() * 100.0
    return mean_p_c, mean_r_c, mean_f_c, precision_o.item(), recall_o.item(), f1_o.item()


def resize_tensor(imgs, target_size = (224, 224), method = 'stretch'):
    resized_imgs = []
    size_flag = torch.zeros(len(imgs))
    if method == 'stretch':
        for i, img in enumerate(imgs):
            if img.size(1) < target_size[0] or img.size(2) < target_size[1]:
                size_flag[i] = 1
            resized_img = F.interpolate(
                img.unsqueeze(0), size = target_size, mode = 'bilinear', align_corners = False
            ).squeeze(0)
            resized_imgs.append(resized_img)
        resized_imgs = torch.stack(resized_imgs, dim=0)
        return resized_imgs, size_flag

    if method == 'padding':
        for i, img in enumerate(imgs):
            if img.size(1) < target_size[0] or img.size(2) < target_size[1]:
                size_flag[i] = 1
            c, h, w = img.shape
            pad_h = target_size[0] - h
            pad_w = target_size[1] - w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            resized_img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            resized_imgs.append(resized_img)
        resized_imgs = torch.stack(resized_imgs, dim=0)
        return resized_imgs, size_flag

    else:
        print("Not valid resize method")
        exit(-1)


class targetCounter:
    def __init__(self) -> None:
        self.num_targets = 0
        self.num_images = 0
        
    def add(self, num_targets=0, num_images=0):
        self.num_targets += num_targets
        self.num_images += num_images
    
    def getAvg(self):
        return self.num_targets / self.num_images
