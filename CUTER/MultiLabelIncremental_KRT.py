import math
from copy import copy, deepcopy
import os
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay, \
    reduce_tensor, AverageMeter
from src.models import my_create_model
from src.loss_functions.losses import AsymmetricLoss
from src.loss_functions.distillation import pod, embeddings_similarity
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
from src.helper_functions.coco_loader import COCOLoader, coco_ids_to_cats, coco_fake2real
from src.helper_functions.IncrementalDataset import build_dataset, build_loader
from sample_proto import icarl_sample_protos, random_sample_protos
from src.helper_functions.utils import build_logger, build_writer

from MultiLabelIncremental_protoConflict import MultiLabelIncremental


class MultiLabelIncremental_KRT(MultiLabelIncremental):
    """
    MultiLabelIncremental For Class Attention
    """
    # Resume, add tokens by steps
    # 4 tokens for base stage
    def setup_model(self):
        """
        Create model
        Load Checkpoint from resume or pretrained weight
        """

        # Resume from checkpoint
        if self.resume:
            model = my_create_model(self.args, num_classes=self.base_classes)
            # For CA increase tokens
            for i, _ in enumerate(range(self.base_classes, self.low_range, self.task_size)):
                model.add_model(self.task_size)
            model = model.cuda()
            state = torch.load(self.resume, map_location='cpu')
            filtered_dict = {k: v for k, v in state.items() if (k in model.state_dict())}  # only for counts
            model.load_state_dict(filtered_dict, strict=False)
            self.logger.info('Create Model successfully, Loaded from resume path:{}, Loaded params:{}\n'
                             .format(self.resume, len(filtered_dict)))

            if self.kd_loss or self.pseudo_label:
                self.old_model = deepcopy(model).eval().cuda()



        # Load pretrained weights
        elif self.pretrained_path:  # make sure to load pretrained ImageNet model
            model = my_create_model(self.args, num_classes=self.base_classes)
            model = model.cuda()
            state = torch.load(self.pretrained_path, map_location='cpu')

            # remove 'body' in params name
            if '21k' in self.pretrained_path:
                state = {(k if 'body.' not in k else k[5:]): v for k, v in state['model'].items()}
                filtered_dict = {k: v for k, v in state.items() if
                                (k in model.state_dict() and 'head.fc' not in k)}
            else:
                state = {(k if 'body.' not in k else k[5:]): v for k, v in state['model'].items()}
                filtered_dict = {k: v for k, v in state.items() if
                                (k in model.state_dict() and 'head.fc' not in k)}

            model.load_state_dict(filtered_dict, strict=False)
            self.logger.info('Create Model successfully, Loaded from model_path:{}, Loaded params:{}\n'
                             .format(self.pretrained_path, len(filtered_dict)))

        return model

    def Dynamic_Pseudo_Labal(self, train_loader, low_range):
        if self.dataset_name == 'voc':
            print("dataset==voc")
            target_label_num = 1.4 * (self.num_classes - self.task_size) / 20
        else:
            target_label_num = 2.9 * (self.num_classes - self.task_size) / 80
        #
        start_time = time.time()
        if (self.pseudo_label or self.kd_loss) and self.old_model is not None:

            old_logits_list = []
            targets_list = []
            for i, (image, target) in enumerate(train_loader):
                image = image.cuda(non_blocking=True)
                with torch.no_grad():
                    old_output = self.old_model(image)
                    old_logits_list.append(old_output['logits'])
                    targets_list.append(target)

            old_logits = torch.concat(old_logits_list)
            targets = torch.concat(targets_list)

            _thre = self.thre
            times = 0
            while times < 100:
                old_sigmoid = torch.sigmoid(old_logits)

                num_pseudo_avg = (old_sigmoid > _thre).sum() / old_sigmoid.shape[0]
                self.logger.info('num_pseudo_avg{:.2f}'.format(num_pseudo_avg))
                if num_pseudo_avg > target_label_num + 0.1:
                    _thre = min(0.999999, _thre + 0.005)
                elif num_pseudo_avg < target_label_num - 0.1:
                    _thre = max(0.0001, _thre - 0.005)
                else:
                    self.logger.info('Calculate pseudo Label Thre time:{:.2f}'.format(time.time() - start_time))
                    self.logger.info('Calculate Thre:{:.2f}'.format(_thre))
                    self.logger.info('Calculate Average pseudo label:{:.2f}'.format(num_pseudo_avg))
                    self.logger.info('Current Avg target num:{:.2f}'.format(targets.sum() / targets.shape[0]))

                    return _thre

                times += 1
            raise Exception(f'Error! Calculate Pseudo Label timeout. Current thre:{_thre}, \
                            Average Numeber of PseudoLabel:{num_pseudo_avg}')
    # Flat loss computation
    def compute_loss(self, output, target,pre_output, low_range, high_range, old_output=None):
        """
        Input: outputs of network, ground truth, Input image(optional, for distilation loss)
        1. classification loss
        2. distillation loss(spatial and flat)
        """

        loss = 0.

        logits = output['logits'].float()
    
        # Classification Loss
     
        cls_loss = self.cls_criterion(logits, target)
    
        # token Loss


        tl_loss = torch.zeros(1).cuda()

        if self.old_model:

            lambda_f_TDL = self.lambda_f_TDL * math.sqrt(self.num_classes / self.task_size)
            new_flat = torch.cat(output['tokens'][:-1], dim=1)  # only old tokens for kd
            old_flat = torch.cat(old_output['tokens'], dim=1)
            tl_loss = embeddings_similarity(old_flat, new_flat)
            tl_loss = lambda_f_TDL * tl_loss



        loss = cls_loss + tl_loss
        return loss, cls_loss, tl_loss

    def _train_one_epoch(self, train_loader, scaler, low_range, high_range, epoch):
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        # Calculate Dynamic Pseudo Label Thre
        if self.pseudo_label and self.old_model and epoch == 0:
            if self.dynamic:
                self.thre = self.Dynamic_Pseudo_Labal(train_loader, low_range)
                self.logger.info('After Calculate thre:{}'.format(self.thre))
            else:
                self.logger.info('Use default thre:{}'.format(self.thre))

        for i, (image, target) in enumerate(train_loader):
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)  # (batch,3,num_classes)
            target = target[:, :high_range]

            # ----------
            # generate pseudo label, compute old output for Knowledge Distillation
            # ----------
            old_output = None
            pre_output = None

            if (self.pseudo_label or self.kd_loss) and self.old_model is not None:
                with torch.no_grad():
                    old_output = self.old_model(image)
                    if self.pseudo_label:
                        old_logits = old_output['logits']
                        old_logits = torch.sigmoid(old_logits)
                        # todo faster
                        old_logits = old_logits.detach()
                        ## adding pseudo-label
                        target[:, :low_range][old_logits > self.thre] =1



            with autocast():  # mixed precision
                output = self.model(image)  # sigmoid will be done in loss !

            loss, cls_loss, token_loss = self.compute_loss(output, target, pre_output, low_range,high_range, old_output=old_output)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            # self.scheduler.step()
            self.model.zero_grad(set_to_none=True)

            # reduce loss for distributed
            if self.world_size > 1:
                loss = reduce_tensor(loss.data, self.world_size)
                cls_loss = reduce_tensor(cls_loss.data, self.world_size)
                if token_loss:
                    token_loss = reduce_tensor(token_loss.data, self.world_size)
            

            # Log trainning information
            if i % self.log_frequency == 0:
                if self.kd_loss == 'None' or self.old_model is None:
                    token_loss = torch.zeros(1)

                self.logger.info(
                    'Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}, cls_loss:{:.1f}, token_loss:{:.1f}'
                        .format(epoch + 1, self.nb_epochs, str(i).zfill(3), str(len(train_loader)).zfill(3),
                                self.scheduler.get_last_lr()[0],
                                loss.item(), cls_loss.item(), token_loss.item()))
        if self.rank == 0:
            self.writer.add_scalar(f"Loss/stage_{low_range}to{high_range}", loss.item(), global_step=epoch)

    # 'head.fc.weight' VS 'head'
    def _finetune_fc(self, low_range, high_range, val_dataset_seen):
        self.logger.info('Starting Finetune FC, Current Incremental Stage is {} to {}, old_dataset length is: {}.'
                         .format(low_range, high_range, len(self.old_dataset)))

        # fix layers
        for k, v in self.model_without_ddp.named_parameters():
            if 'head' not in k:
                v.requires_grad = False

        # Get old datasets loader
        proto_ds = torch.utils.data.ConcatDataset(self.old_dataset)
        protos_loader = build_loader(proto_ds, self.batch_size, self.num_workers, phase='train')

        optimizer = torch.optim.Adam(self.model_without_ddp.head.parameters(), lr=self.lr/2,
                                     weight_decay=self.weight_decay)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr/2, steps_per_epoch=len(protos_loader),
                                            epochs=self.nb_epochs, pct_start=0.2)
        scaler = GradScaler()

        self.model.train()
        for epoch in range(0, self.nb_epochs):
            if epoch == self.end_epoch:
                break

            protos_loader.sampler.set_epoch(epoch)

            self.model.zero_grad(set_to_none=True)
            for i, (images, targets) in enumerate(protos_loader):
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)  # (batch,3,num_classes)

                targets = targets[:, :high_range]
                with autocast():  # mixed precision
                    output = self.model(images)  # sigmoid will be done in loss !

                logits = output['logits']
                loss = self.cls_criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                self.model.zero_grad(set_to_none=True)

                if epoch % 5 == 0 and i % 50 == 0:
                    self.logger.info('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                                     .format(epoch + 1, self.nb_epochs, str(i).zfill(3),
                                             str(len(protos_loader)).zfill(3),
                                             scheduler.get_last_lr()[0],
                                             loss.item()))

        # Validate at lasttmux
        val_loader_seen = build_loader(val_dataset_seen, self.batch_size, self.num_workers, phase='val')
        val_result = self.validate(low_range, high_range, None, val_loader_seen, None, only_seen=True)
        self.logger.info('Finetune FC current_mAP_seen = {:.2f}'.format(val_result['seen'][0]))

        if self.rank == 0:
            self.writer.add_scalar("mAP_ftfc/seen", val_result['seen'][0], global_step=high_range)

        # unfreeze params
        for k, v in self.model_without_ddp.named_parameters():
            if 'head' not in k:
                v.requires_grad = True

        return val_result['seen'][0]

    # Add classes
    def _before_task(self, low_range, high_range):

        self.model.eval()
        self.num_classes = high_range
      #  self.class_mask = [*range(low_range, high_range)]


 
        if low_range > 0:
            self.model_without_ddp.add_model(self.task_size)
            self.model_without_ddp.freeze(self.args.freeze_task)

        new_token_data = self.model_without_ddp.task_tokens[-1].data
        dist.broadcast(new_token_data, 0)

        if low_range ==0:
           new_cls_data = self.model_without_ddp.cls_token[-1].data
           dist.broadcast(new_cls_data, 0)


        # todo change the included cats of old datasets, due to proto Conflict
        if self.old_dataset:
            if self.args.fix_budget:
                self.old_dataset.update_cls(self.num_classes)
            else:
                for subset in self.old_dataset:
                    dataset = subset.dataset
                    old_low_range = dataset.included_cats[0]
                    new_retrieve_classes = range(old_low_range, self.num_classes)
                    dataset.included_cats = new_retrieve_classes

