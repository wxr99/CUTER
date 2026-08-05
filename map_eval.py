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

fiedler_list = []


class MultiLabelEval:
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
            state = {(k): v for k, v in model.named_parameters()}
            filtered_dict = {k: v for k, v in model.named_parameters() if not k.startswith('fc')}
            self.logger.info('Create Model successfully, Loaded from torchvision 21k pre-trained model, Loaded params:{}\n'
                             .format(sum(p.numel() for p in filtered_dict.values())))
        return model

    def _before_task(self, low_range, high_range):

        self.model.eval()
        self.num_classes = high_range

    def _train_one_epoch(self, train_loader, scaler, low_range, high_range, epoch):
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        for i, (image, target) in enumerate(train_loader):
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # ----------
            # forward
            # ----------
            with torch.no_grad():  # mixed precision
                output_list = self.model(image, map=True)
                maps = output_list['maps']
                maps = maps.transpose(1, 2)
                normalized_maps = F.normalize(maps, p=2, dim=2)
                dot = torch.bmm(normalized_maps, normalized_maps.transpose(1,2))
                dis = torch.clamp((2-2*dot), min=0.0)
                sim=torch.exp(-dis/(2*1**2))

                for adj in sim:
                    degrees = torch.sum(adj, dim=1)
                    D = torch.diag(degrees)
                    L = D - adj
                    eigenvals = torch.linalg.eigvalsh(L)
                    fiedler = eigenvals[1].real
                    # print(fiedler)
                    fiedler_list.append(fiedler)


    def _train_task(self, initial_epoch, nb_epochs, low_range, high_range, train_loader):
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


            # Build loaders
            train_loader = build_loader(train_dataset, self.batch_size, self.num_workers, phase='train')

            # ----------
            # Training process
            # ----------

            self._before_task(low_range, high_range)
            self._train_task(0, self.nb_epochs, low_range, high_range, train_loader)
        sum=0
        for v in fiedler_list:
            sum += v
        print(sum/len(fiedler_list))
        del self.logger
