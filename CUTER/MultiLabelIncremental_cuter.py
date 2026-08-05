import math
from copy import copy, deepcopy
import os
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from MultiLabelIncremental_protoConflict import buffer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay, \
    reduce_tensor, AverageMeter
from src.models import my_create_model
from src.loss_functions.losses import AsymmetricLoss
from src.loss_functions.distillation import pod, embeddings_similarity
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from src.helper_functions.coco_loader import COCOLoader, coco_ids_to_cats, coco_fake2real
from src.helper_functions.IncrementalDataset import build_dataset, build_loader
from src.helper_functions.utils import resize_tensor
from sample_proto import icarl_sample_protos, random_sample_protos, icarl_sample_protos_buffer
from src.helper_functions.utils import build_logger, build_writer, print_to_excel, calculate_metrics
from src.helper_functions import maskcut
from git import Repo


class RS_Buffer():
    def __init__(self, buffer_size) -> None:
        self.buffer = torch.tensor([]).cuda()
        self.label = torch.tensor([]).cuda()
        self.budget_size = buffer_size

    def add(self, x, y, total_num, low_range, high_range):
        mask = torch.ones_like(y)
        mask[:, low_range:high_range] = 0
        y = y -  mask.cuda()

        if len(self.buffer) == 0:
            self.buffer = x
            self.label = y
        elif len(self.buffer) + len(x) <= self.budget_size:
            self.buffer = torch.cat((self.buffer, x))
            self.label = torch.cat((self.label, y))
        else:
            for i, l in enumerate(y):
                p = random.random()
                if p <= 1 / total_num:
                    index = random.randint(0, self.budget_size-1)
                    mask = torch.ones_like(l)
                    mask[low_range:high_range] = 0
                    self.label[index] = l - mask.cuda()
                    self.buffer[index] = x[i]

# Reservoir Sampling
class Cut_Balanced_Buffer():
    def __init__(self, buffer_size, total_classes) -> None:
        self.buffer = []
        self.hot_label = torch.tensor([]).cuda()
        self.budget_size = buffer_size
        self.total_classes = total_classes
        self.class_freq = torch.zeros(self.total_classes).cuda()
        self.class_flag = torch.zeros(self.total_classes).cuda()

    def add(self, x, y, current_classes, low_range, high_range):
        indices = torch.arange(len(self.class_flag)).cuda()
        class_budget = self.budget_size // current_classes
        flag_threshold = self.class_freq.max() // 2
        hot_labels = F.one_hot(y, self.total_classes)
        # print("Buffer Allocation:", self.class_freq)
        if len(self.buffer) == 0:
            self.buffer = x
            self.hot_label = hot_labels
            flag_index = torch.where(self.class_freq < flag_threshold)[0]
            self.class_flag[flag_index] = 1
            self.class_flag = torch.where((indices>=low_range)&(indices<high_range), self.class_flag, torch.tensor(0.0).cuda())
            self.class_freq += torch.sum(hot_labels, dim=0)
        else:
            for i, label in enumerate(y):
                if len(self.buffer) < self.budget_size:
                    if self.class_freq[label] < class_budget:
                        self.buffer.append(x[i])
                        self.hot_label = torch.cat([self.hot_label, hot_labels[i].unsqueeze(0)], dim=0)
                        self.class_freq[label] += 1
                    else:
                        index = torch.where(self.hot_label[:, label] == 1)[0]
                        index = index[torch.randint(len(index), (1,)).item()]
                        self.buffer[index] = x[i]
                else:
                    p = random.random()
                    max_num, max_class = self.class_freq.max(dim=0)
                    if p > self.class_freq[label] / max_num:
                        index = torch.where(self.hot_label[:, max_class] == 1)[0]
                        index = index[torch.randint(len(index), (1,)).item()]
                        self.buffer[index] = x[i]
                        self.hot_label[index] = hot_labels[i]
                        self.class_freq[max_class] -= 1
                        self.class_freq[label] += 1


            flag_index = torch.where(self.class_freq < flag_threshold)[0]
            self.class_flag = torch.zeros(self.total_classes).cuda()
            self.class_flag[flag_index] = 1
            self.class_flag = torch.where((indices >= low_range) & (indices < high_range), self.class_flag,
                                          torch.tensor(0.0).cuda())




class MultiLabelIncremental_CUTER:
    def __init__(self, args):
        # self.logger.info('Running CUTER')

        self.args = args

        # Distributed
        self.world_size = args.world_size
        self.rank = args.local_rank
        self.distillation_tau = 1
        # Output
        self.log_frequency = 100
        self.logger = build_logger(args.logger_dir, self.rank)
        self.logger.info('Arguments:')
        for k, v in sorted(vars(args).items()):
            self.logger.info('{}={}'.format(k, v))

        if self.rank == 0:
            self.writer = build_writer(args.tensorboard_dir, self.args)
        self.model_save_path = args.model_save_path
        if not os.path.exists(self.model_save_path) and self.rank == 0:
            os.makedirs(self.model_save_path)

        self.excel_path = args.excel_path

        # Train params
        self.nb_epochs = args.epochs
        self.end_epoch = args.end_epoch
        # todo test for lr
        # self.lr = (args.batch_size * args.world_size) / 256 * args.lr
        self.incr_lr = args.lr
        self.base_lr = args.base_lr
        self.weight_decay = args.weight_decay

        # Incremental
        self.base_classes = args.base_classes
        self.task_size = args.task_size
        self.total_classes = args.total_classes

        self.num_classes = args.base_classes
        # model
        self.model_name = args.model_name
        self.pretrained_path = args.pretrained_path

        # resume
        self.resume = args.resume
        self.old_dataset_path = args.old_dataset_path
        self.low_range = args.low_range
        self.alpha = args.alpha
        self.beta = args.beta

        self.old_model = None

        # datasets
        self.dataset_name = args.dataset_name
        self.root_dir = args.root_dir

        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.mem_batch_size = args.mem_size
        # self.num_workers = args.num_workers
        # todo faster
        self.num_workers = args.num_workers
        self.train_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
            # normalize,
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            # normalize, # no need, toTensor does normalization
        ])

        # replay
        self.replay = args.replay
        self.total_num = 0
        self.num_protos = args.num_protos
        if self.replay:
            self.mem_buffer = Cut_Balanced_Buffer(args.buffer_size, args.total_classes)
        self.temperature = args.temperature

        # todo old_dataset is a List of dataset
        self.old_dataset = []
        self.image_id_set = set()
        self.sample_method = args.sample_method
        self.kd_loss = False

        # pseudo label
        self.pseudo_label = args.pseudo_label
        self.dynamic = args.dynamic if 'dynamic' in args else True
        self.thre = args.thre
        self.gt_num = []
        self.class_mask = None

        # finetune fc
        self.finetune_fc = args.finetune_fc

        # model
        self.CLASSES_PER_TOKEN = self.task_size
        self.model = self.setup_model()
        self.model_without_ddp = self.model
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank],
                                                               broadcast_buffers=False,
                                                               find_unused_parameters=True)

        # todo faster
        torch.backends.cudnn.benchmark = True

        if self.rank == 0:
            self.logger.info("Arguments:")
            for k, v in sorted(vars(args).items()):
                self.logger.info('{} = {}'.format(k, v))

    def setup_model(self):
        """
        Create model
        Load Checkpoint from resume or pretrained weight
        """

        # Resume from checkpoint
        if self.resume:
            # model = my_create_model(self.args, self.base_classes)
            model = my_create_model(self.args, self.total_classes)
            model = model.cuda()
            state = torch.load(self.resume, map_location='cpu')
            filtered_dict = {k: v for k, v in state.items() if (k in model.state_dict())}  # only for counts
            model.load_state_dict(filtered_dict, strict=False)
            self.logger.info('Create Model successfully, Loaded from resume path:{}, Loaded params:{}\n'
                             .format(self.resume, len(filtered_dict)))

        # Load pretrained weights
        elif self.pretrained_path:  # make sure to load pretrained ImageNet model
            model = my_create_model(self.args, self.total_classes)
            model = model.cuda()
            # state = {(k): v for k, v in model.named_parameters()}
            filtered_dict = {k: v for k, v in model.named_parameters() if not k.startswith('fc')}
            self.logger.info('Create Model successfully, Loaded from torchvision 21k pre-trained model, Loaded params:{}\n'
                             .format(sum(p.numel() for p in filtered_dict.values())))
        return model

    def compute_loss(self, output, target, low_range=None, high_range=None):
        """
        Input: outputs of network, ground truth, Input image(optional, for distilation loss)
        1. classification loss
        2. distillation loss(spatial and flat)
        """
        logits = output.float()
        # Classification Loss
        cls_loss = self.cls_criterion(logits, target, low_range, high_range)
        loss = cls_loss
        return loss, cls_loss

    def _before_task(self, low_range, high_range):

        self.model.eval()
        self.num_classes = high_range

    def _train_one_epoch(self, train_loader, scaler, low_range, high_range, epoch):
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        for i, (image, target) in enumerate(train_loader):
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # print(torch.where(target==1))
            self.total_num += len(target)

            # ----------
            # forward
            # ----------
            with autocast():  # mixed precision
                output_list = self.model(image, map=True)
                output = output_list['logits']   # sigmoid will be done in loss !
                maps = output_list['maps']
                bz = len(output)

                cropped_img_list = maskcut.maskcut(imgs=image,maps=maps.detach(), h=224, w=224, patch_size=16, tau=0.2, N=2, cpu=False)
                crop_image_1, size_flag = resize_tensor(imgs=cropped_img_list, target_size=(224, 224), method='padding')
                # crop_image_2, _ = resize_tensor(imgs=cropped_img_list, target_size=(224, 224), method='stretch')

                maps = maps.transpose(1, 2)
                normalized_maps = F.normalize(maps, p=2, dim=2)
                dot = torch.bmm(normalized_maps, normalized_maps.transpose(1, 2))
                dis = torch.clamp((2 - 2 * dot), min=0.0)
                sim = torch.exp(-dis / (2 * self.temperature ** 2))
                # sim = torch.bmm(normalized_maps, normalized_maps.transpose(1, 2)) / self.temperature
                nuclear_norms = torch.linalg.matrix_norm(sim.float(), ord='nuc', dim=(-2, -1))
                l1_norms = torch.norm(sim.reshape(sim.shape[0], -1), p=1, dim=1)
                l2_norms = torch.norm(sim.reshape(sim.shape[0], -1), p=1, dim=1)
                reg = self.alpha * nuclear_norms.sum() + self.beta *l1_norms.sum() / l2_norms.sum()

            with torch.no_grad():
                crop_output_1 = self.model(crop_image_1)['logits']
                # crop_output_2 = self.model(crop_image_2)['logits']
                crop_pred_1 = torch.sigmoid(crop_output_1)
                # crop_pred_2 = torch.sigmoid(crop_output_2)
                pseudo_label1 = torch.argmax(crop_pred_1, dim=1)
                # pseudo_label2 = torch.argmax(crop_pred_2, dim=1)
                # mask = pseudo_label1 == pseudo_label2

                rep_target = target.repeat_interleave(2, dim=0)
                flag_class = torch.where(self.mem_buffer.class_flag == 1)[0]
                # print('Scarse class:', self.mem_buffer.class_flag)
                balance_index = torch.where(rep_target[:, flag_class]==1)[0]
                delete_index = torch.where(crop_pred_1[:, flag_class] < 0.55)[0]
                mask = ~torch.isin(balance_index, delete_index)
                balance_index = balance_index[mask]

                if len(balance_index) > 1:
                    balance_hot = rep_target[balance_index]
                    # print('Index:', balance_index)
                    balance_imgs = [cropped_img_list[i] for i in balance_index.tolist()]
                    bool_mask = torch.zeros(self.total_classes, dtype=bool)
                    bool_mask[flag_class] = True
                    balance_hot[:, ~bool_mask] = 0
                    probs = balance_hot / balance_hot.sum(dim=1, keepdim=True)
                    balance_targets = torch.multinomial(probs, num_samples=1).squeeze()
                    self.mem_buffer.add(balance_imgs, balance_targets, high_range - low_range, low_range, high_range)


                mem_flag = rep_target[torch.arange(len(rep_target)), pseudo_label1] * size_flag.cuda()
                mem_index = torch.where(mem_flag==1)[0]
                mask = ~torch.isin(mem_index, balance_index)
                mem_index = mem_index[mask]



                if len(mem_index) > 1:
                    mem_imgs = [cropped_img_list[i] for i in mem_index.tolist()]
                    mem_targets = pseudo_label1[mem_index]
                    # print(mem_targets)
                    self.mem_buffer.add(mem_imgs, mem_targets, high_range-low_range, low_range, high_range)



            _, cls_loss = self.compute_loss(output, target, low_range, high_range)
            loss = cls_loss + reg

            if len(self.mem_buffer.buffer) > self.mem_batch_size:
                mem_dataset, _ = resize_tensor(imgs=self.mem_buffer.buffer, target_size=(224, 224), method='padding')
                memory = TensorDataset(mem_dataset, self.mem_buffer.hot_label)
                memory_loader = DataLoader(dataset=memory, batch_size=self.batch_size, shuffle=True)
                mem_data = iter(memory_loader)
                mem_img, mem_targets = next(mem_data)
                mem_output_list = self.model(mem_img)
                mem_output = mem_output_list['logits']
                # print(mem_targets)
                # mem_loss = F.cross_entropy(mem_output, mem_targets.float())
                mem_loss, _ = self.compute_loss(mem_output, mem_targets)
                loss += mem_loss

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.model.zero_grad(set_to_none=True)

            # reduce loss for distributed
            if self.world_size > 1:
                loss = reduce_tensor(loss.data, self.world_size)
                cls_loss = reduce_tensor(cls_loss.data, self.world_size)

            # Log trainning information
            if i % self.log_frequency == 0:
                self.logger.info(
                    'Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}, cls_loss:{:.1f}'
                    .format(epoch + 1, self.nb_epochs, str(i).zfill(3), str(len(train_loader)).zfill(3),
                            self.lr, loss.item(), cls_loss.item()))
        if self.rank == 0:
            self.writer.add_scalar(f"Loss/stage_{low_range}to{high_range}", loss.item(), global_step=epoch)

    def _train_task(self, initial_epoch, nb_epochs, low_range, high_range, train_loader,
                    val_loader_base, val_loader_seen, val_loader_new):
        self.model.train()
        # Use different lr for base classes

        if low_range == 0:
            self.lr = self.base_lr
        else:
            self.lr = self.incr_lr

        parameters = add_weight_decay(self.model_without_ddp, self.weight_decay)
        self.cls_criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
        self.optimizer = torch.optim.Adam(params=parameters, lr=self.lr, weight_decay=0)  # true wd, filter_bias_and_bn

        scaler = GradScaler()
        for epoch in range(initial_epoch, nb_epochs):
            # Early Stop with 80 epochs
            if epoch == self.end_epoch:
                break

            train_loader.sampler.set_epoch(epoch)

            epoch_start = time.time()
            self._train_one_epoch(train_loader, scaler, low_range, high_range, epoch)
            train_epoch_time = time.time() - epoch_start

            self.model.eval()
            val_start = time.time()
            val_result, val_result2 = self.validate(low_range, high_range, val_loader_base, val_loader_seen,
                                                    val_loader_new,
                                                    only_seen=(low_range == 0))
            val_time = time.time() - val_start

            self.logger.info('Train one epoch time:{:.2f}, validate time:{:.2f}.'.format(train_epoch_time, val_time))

            # Base stage
            if low_range == 0:
                val_result['base'] = val_result['seen']
                val_result['new'] = val_result['seen']
            if self.args.only_seen:
                val_result['base'] = val_result['seen']
                val_result['new'] = val_result['seen']
            self.logger.info('current_mAP_base = {:.2f}'.format(val_result['base'][0]))
            self.logger.info('current_mAP_seen = {:.2f}'.format(val_result['seen'][0]))
            self.logger.info('current_mAP_new = {:.2f}'.format(val_result['new'][0]))
            if self.rank == 0:
                self.logger.info(
                    'current other metrics:, mean_p_c:{}, mean_r_c:{}, mean_f_c:{}, precision_o:{}, recall_o:{}, f1_o:{}'
                    .format(*[i for i in val_result2['seen']]))

            # Tensorboard, record the map per epoch
            # To achieve a balance between old and new classes
            if self.rank == 0:
                self.writer.add_scalar("Stage_{}to{}/mAP per epoch/base".format(low_range, high_range),
                                       val_result['base'][0], global_step=epoch)
                self.writer.add_scalar("Stage_{}to{}/mAP per epoch/seen".format(low_range, high_range),
                                       val_result['seen'][0], global_step=epoch)
                self.writer.add_scalar("Stage_{}to{}/mAP per epoch/new".format(low_range, high_range),
                                       val_result['new'][0], global_step=epoch)

        # 用最后一个epoch输出的map作为该阶段的指标
        if self.rank == 0:
            self.writer.add_scalar("mAP/base", val_result['base'][0], global_step=high_range)
            self.writer.add_scalar("mAP/seen", val_result['seen'][0], global_step=high_range)
            self.writer.add_scalar("mAP/new", val_result['new'][0], global_step=high_range)
            # for cls, score in enumerate(val_result['seen'][1]):
            #     real_class = coco_fake2real[cls]
            #     real_class = coco_ids_to_cats[real_class]
            #     self.writer.add_scalar("AP/{}".format(real_class), score, global_step=high_range)

        return val_result['seen'][0], val_result2['seen']

    def _after_task(self, low_range, high_range, train_dataset=None):
        """
        保存当前阶段的model和proto
        """
        # ----------
        # Save old model
        # ----------
        if self.rank == 0:
            torch.save(self.model_without_ddp.state_dict(), '{}/{}_{}_{}to{}.pth'.format(
                self.model_save_path, self.dataset_name, self.model_name, low_range, high_range))


    def validate(self, low_range, high_range, val_loader_base, val_loader_seen, val_loader_new, only_seen=False):
        """
        多卡验证，分别在基类，所有类，新类上进行验证
        """
        self.model.eval()
        self.logger.info("Starting validation")
        Sig = torch.nn.Sigmoid()

        val_result = {}
        val_result2 = {}
        val_result2['seen'] = [0]
        only_seen = self.args.only_seen
        # validate on 3 datasets
        val_stage = ['base', 'seen', 'new']
        val_classes = [(0, self.base_classes), (0, high_range), (low_range, high_range)]
        for i, val_loader in enumerate([val_loader_base, val_loader_seen, val_loader_new]):
            # Todo "only seen" is for debug, delete later
            if only_seen and val_stage[i] != 'seen':
                continue

            preds_regular = []
            targets = []

            for image, target in val_loader:
                # todo faster
                image = image.cuda()
                target = target[:, val_classes[i][0]:val_classes[i][1]].cuda()

                with torch.no_grad():
                    with autocast():
                        output_regular = Sig(self.model(image)['logits'])
                        output_regular = output_regular[:, val_classes[i][0]:val_classes[i][1]].contiguous()

                # 将不同GPU上的结果汇总
                if self.world_size > 1:
                    output_gather_list = [torch.zeros_like(output_regular) for _ in range(self.world_size)]
                    target_gather_list = [torch.zeros_like(target) for _ in range(self.world_size)]

                    dist.all_gather(output_gather_list, output_regular)
                    dist.all_gather(target_gather_list, target)

                    output_regular = torch.cat(output_gather_list, dim=0)
                    target = torch.cat(target_gather_list, dim=0)

                # for mAP calculation
                # todo faster
                preds_regular.append(output_regular.detach())
                targets.append(target.detach())

            mAP_score_regular = 0
            score_regular = 0
            mean_p_c = 0
            mean_r_c = 0
            mean_f_c = 0
            precision_o = 0
            recall_o = 0
            f1_o = 0

            if self.rank == 0:
                mAP_score_regular, score_regular = mAP(torch.cat(targets).cpu().numpy(),
                                                       torch.cat(preds_regular).cpu().numpy())

                if val_stage[i] == 'seen':
                    print('calculate metrics')
                    mean_p_c, mean_r_c, mean_f_c, precision_o, recall_o, f1_o = calculate_metrics(
                        torch.cat(preds_regular).cpu(), torch.cat(targets).cpu(), thre=0.8)
                    val_result2['seen'] = [mean_p_c, mean_r_c, mean_f_c, precision_o, recall_o, f1_o]

            val_result[val_stage[i]] = (mAP_score_regular, score_regular)

        return val_result, val_result2

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

        optimizer = torch.optim.Adam(self.model_without_ddp.head.fc.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=len(protos_loader),
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

                logits = output
                loss = self.cls_criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # scheduler.step()
                self.model.zero_grad(set_to_none=True)

                if epoch % 5 == 0 and i % 50 == 0:
                    self.logger.info('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                                     .format(epoch + 1, self.nb_epochs, str(i).zfill(3),
                                             str(len(protos_loader)).zfill(3),
                                             scheduler.get_last_lr()[0],
                                             loss.item()))

        # Validate at lasttmux
        val_loader_seen = build_loader(val_dataset_seen, self.batch_size, self.num_workers, phase='val')
        val_result, val_result2 = self.validate(low_range, high_range, None, val_loader_seen, None, only_seen=True)
        self.logger.info('Finetune FC current_mAP_seen = {:.2f}'.format(val_result['seen'][0]))

        if self.rank == 0:
            self.writer.add_scalar("mAP_ftfc/seen", val_result['seen'][0], global_step=high_range)
            self.logger.info(
                'current other metrics:, mean_p_c:{}, mean_r_c:{}, mean_f_c:{}, precision_o:{}, recall_o:{}, f1_o:{}'
                .format(*[i for i in val_result2['seen']]))
        # unfreeze params
        for k, v in self.model_without_ddp.named_parameters():
            if 'head' not in k:
                v.requires_grad = True

        return val_result['seen'][0]

    def train(self):
        mAP_meter = AverageMeter()

        mAP_list = np.zeros((self.total_classes - self.base_classes) // self.task_size + 1)
        cf1_list = np.zeros((self.total_classes - self.base_classes) // self.task_size + 1)
        of1_list = np.zeros((self.total_classes - self.base_classes) // self.task_size + 1)
        if self.finetune_fc:
            mAP_ftfc_meter = AverageMeter()

        if self.args.freeze_bockbone and self.low_range is not None:
            print('########freeze#############')
            for k, v in self.model_without_ddp.named_parameters():
                if 'head' not in k:
                    v.requires_grad = False

        # 设置增量阶段
        if self.resume:
            incremental_stages = [(low, low + self.task_size) for low in
                                  range(self.low_range, self.total_classes, self.task_size)]

            if self.replay:
                self.old_dataset = torch.load(self.old_dataset_path)
        else:
            base_stage = [(0, self.base_classes)]
            incremental_stages = base_stage + [
                (low, low + self.task_size) for low in range(self.base_classes, self.total_classes, self.task_size)]

        for low_range, high_range in incremental_stages:
            # ----------
            # Getting Dataset and Dataloader
            # ----------

            # Build train dataset
            train_dataset = build_dataset(self.dataset_name, self.root_dir, low_range, high_range,
                                                      phase='train', transform=self.train_transforms)
            self.logger.info('Current incremental stage:({},{}), dataset length:{}'
                             .format(low_range, high_range, len(train_dataset)))

            # 3 validation datasets: base, seen, new
            val_dataset_base = build_dataset(self.dataset_name, self.root_dir, 0, self.base_classes, phase='val',
                                             transform=self.val_transforms)
            val_dataset_seen = build_dataset(self.dataset_name, self.root_dir, 0, high_range, phase='val',
                                             transform=self.val_transforms)
            val_dataset_new = build_dataset(self.dataset_name, self.root_dir, low_range, high_range, phase='val',
                                            transform=self.val_transforms)

            # Build loaders
            train_loader = build_loader(train_dataset, self.batch_size, self.num_workers, phase='train')
            val_loader_base = build_loader(val_dataset_base, self.batch_size, self.num_workers, phase='val')
            val_loader_seen = build_loader(val_dataset_seen, self.batch_size, self.num_workers, phase='val')
            val_loader_new = build_loader(val_dataset_new, self.batch_size, self.num_workers, phase='val')

            # ----------
            # Training process
            # ----------

            self._before_task(low_range, high_range)
            mAP, metrics = self._train_task(0, self.nb_epochs, low_range, high_range, train_loader, val_loader_base,
                                            val_loader_seen, val_loader_new)
            mean_p_c, mean_r_c, mean_f_c, precision_o, recall_o, f1_o = metrics[0], metrics[1], metrics[2], metrics[3], \
            metrics[4], metrics[5]
            mAP_meter.update(mAP)
            mAP_list[(high_range - self.base_classes) // self.task_size] = mAP
            of1_list[(high_range - self.base_classes) // self.task_size] = f1_o
            cf1_list[(high_range - self.base_classes) // self.task_size] = mean_f_c

            self._after_task(low_range, high_range, train_dataset=train_dataset)
            if self.finetune_fc:
                if self.old_dataset:
                    mAP_ftfc = self._finetune_fc(low_range, high_range, val_dataset_seen)
                else:
                    mAP_ftfc = mAP  # Copy base stage mAP

                mAP_ftfc_meter.update(mAP_ftfc)
                self.logger.info('mAP_ftfc:{:.2f}'.format(mAP_ftfc))

            if self.rank == 0:
                self.writer.add_scalar('mAP/average', mAP_meter.avg, global_step=high_range)
                if self.finetune_fc:
                    self.writer.add_scalar('mAP_ftfc/average', mAP_ftfc_meter.avg, global_step=high_range)

            if self.args.one_session:
                self.logger.info('End! Only train for 1 session.')
                break

        # TODO print result to excel
        if self.rank == 0:
            if 'coco' in self.dataset_name:
                ds_name = 'COCO'
            if 'voc' in self.dataset_name:
                ds_name = 'VOC'
            if 'nuswide' in self.dataset_name:
                ds_name = 'NUSWIDE'
            expe_name = self.args.output_name
            #   git_hash = Repo(os.getcwd()).git.rev_parse("HEAD")
            params = f"LR:{self.lr}, epoch:{self.end_epoch}, BS:{self.batch_size}, GPU:{self.world_size}, Anchor:{self.num_protos}"
            print_to_excel(self.excel_path, expe_name, ds_name, self.base_classes,
                           self.task_size, self.total_classes, params, mAP_list, cf1_list, of1_list, metrics, git_hash=None)
        self.logger.info('averaged mAP:{}, averaged CF1:{}, averaged oF1:{}'.format(sum(mAP_list)/len(mAP_list), sum(cf1_list)/len(cf1_list), sum(of1_list)/len(of1_list)))
        # so that the logger won't output many times when tuning params
        size = 0
        for frame in self.mem_buffer.buffer:
            d, h, w = frame.shape
            size += d*h*w
        print('Memory Buffer size equals to ', size / (224*224*3))
        del self.logger
