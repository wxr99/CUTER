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
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay, \
    reduce_tensor, AverageMeter
from src.models import my_create_model
from src.loss_functions.losses import AsymmetricLoss
from src.loss_functions.distillation import pod, embeddings_similarity
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
from src.helper_functions.coco_loader import COCOLoader, coco_ids_to_cats, coco_fake2real
from src.helper_functions.IncrementalDataset import build_dataset, build_loader
from sample_proto import icarl_sample_protos, random_sample_protos, icarl_sample_protos_buffer
from src.helper_functions.utils import build_logger, build_writer, print_to_excel, calculate_metrics
from git import Repo

class buffer:
    def __init__(self)-> None:
         self.subsets = []
         self.indexes = []
    def add(self,subset,indexes):

        self.subsets.append(subset)
        self.indexes.append(np.stack(indexes,axis=0))

    def update_cls(self,num_class):
        for subset in self.subsets:
              dataset=subset.dataset
              old_low_range = dataset.included_cats[0]
              new_retrieve_classes = range(old_low_range,num_class)
              dataset.included_cats =new_retrieve_classes

    def get_bufferset(self,buffer_size):
        cls_num =sum(len(indexes) for indexes in self.indexes)
        num_pro = int(np.ceil(buffer_size/cls_num))
        for i,subset in enumerate(self.subsets):
            new_index = self.indexes[i][:,:num_pro]
            subset.indices = np.concatenate(new_index).tolist()
        mem_pro = torch.utils.data.ConcatDataset(self.subsets)
        return mem_pro


class MultiLabelIncremental:
    def __init__(self, args):
        print('Running MLIL protoConflict')

        self.args = args

        # Distributed 
        self.world_size = args.world_size
        self.rank = args.local_rank
        self.distillation_tau = 1
        # Output
        self.log_frequency = 100
        self.logger = build_logger(args.logger_dir, self.rank)
        self.logger.info('Running MLIL protoConflict')
      #  self.logger.info('Git hash:{}'.format(Repo(os.getcwd()).git.rev_parse("HEAD")))
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

        self.num_classes = self.base_classes

        # Knowledge Distillation

        self.kd_loss = args.kd_loss
        if self.kd_loss == 'pod_spatial':
            self.lambda_c = args.lambda_c
            self.lambda_f = args.lambda_f
           
        if self.kd_loss =='token_loss':
            self.lambda_f_TDL = args.lambda_f_TDL if 'lambda_f_TDL' in args else 0
        
        # model
        self.model_name = args.model_name
        self.pretrained_path = args.pretrained_path

        # resume
        self.resume = args.resume
        self.old_dataset_path = args.old_dataset_path
        self.low_range = args.low_range

        self.old_model = None


        # datasets
        self.dataset_name = args.dataset_name
        self.root_dir = args.root_dir

        self.image_size = args.image_size
        self.batch_size = args.batch_size
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
        self.num_protos = args.num_protos
        # todo old_dataset is a List of dataset
        self.old_dataset = []
        self.image_id_set = set()
        self.sample_method = args.sample_method

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
            #model = my_create_model(self.args, self.base_classes)
            model = my_create_model(self.args, self.low_range)
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

            model = my_create_model(self.args, self.base_classes)
            model = model.cuda()
            state = torch.load(self.pretrained_path, map_location='cpu')

            # remove 'body' in params name
            if '21k' in self.pretrained_path:
                # print(state)
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

    def compute_loss(self, output, target, old_output=None):
        """
        Input: outputs of network, ground truth, Input image(optional, for distilation loss)
        1. classification loss
        2. distillation loss(spatial and flat)
        """
        loss = 0.
        logits = output['logits'].float()
        # Classification Loss
        cls_loss = self.cls_criterion(logits, target)

        # KD Loss
        kd_loss = 0.
        pod_spatial = torch.zeros(1)
        pod_flat = torch.zeros(1)
        if self.kd_loss == 'pod_spatial' and self.old_model:
            lambda_c = self.lambda_c * math.sqrt(self.num_classes / self.task_size)
            lambda_f = self.lambda_f * math.sqrt(self.num_classes / self.task_size)

            # Only use the last layer output for distillation loss
            old_features = old_output['attentions']
            new_features = output['attentions']
            pod_spatial = pod(old_features, new_features, 'spatial')

            # Only use flat loss
            pod_flat = embeddings_similarity(old_output['pool_embeddings'], output['pool_embeddings'])

            pod_flat = lambda_f * pod_flat
            pod_spatial = lambda_c * pod_spatial
            kd_loss = pod_flat + pod_spatial

        loss = cls_loss + kd_loss
        return loss, cls_loss, pod_spatial, pod_flat

    def add_classes(self, increment_classes):
        """
        扩大网络FC层 增加新类别 
        """
        in_dimension = self.model_without_ddp.head.fc.in_features
        old_classes = self.model_without_ddp.head.fc.out_features

        new_fc = nn.Linear(in_dimension, old_classes + increment_classes)
        new_fc.weight.data[:old_classes] = self.model_without_ddp.head.fc.weight.data
        new_fc.bias.data[:old_classes] = self.model_without_ddp.head.fc.bias.data
        new_fc.cuda()

        self.model_without_ddp.head.fc = new_fc

    def _before_task(self, low_range, high_range):

        self.model.eval()
        self.num_classes = high_range
        if low_range != 0:
            self.add_classes(self.task_size)

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


    def _train_one_epoch(self, train_loader, scaler, low_range, high_range, epoch):
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        for i, (image, target) in enumerate(train_loader):
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)  # (batch,3,num_classes)
            target = target[:, :high_range]


            old_output = None
            if self.kd_loss and self.old_model is not None:
                with torch.no_grad():
                    old_output = self.old_model(image)
            # ----------
            # forward
            # ----------

            with autocast():  # mixed precision
                output = self.model(image)  # sigmoid will be done in loss !

            loss, cls_loss, pod_spatial, pod_flat = self.compute_loss(output, target, old_output=old_output)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.scheduler.step()
            self.model.zero_grad(set_to_none=True)

            # reduce loss for distributed
            if self.world_size > 1:
                loss = reduce_tensor(loss.data, self.world_size)
                cls_loss = reduce_tensor(cls_loss.data, self.world_size)
                if pod_spatial:
                    pod_spatial = reduce_tensor(pod_spatial.data, self.world_size)
                if pod_flat:
                    pod_flat = reduce_tensor(pod_flat.data, self.world_size)

            # Log trainning information
            if i % self.log_frequency == 0:
                if self.kd_loss == 'None' or self.old_model is None:
                    pod_spatial = torch.zeros(1)
                    pod_flat = torch.zeros(1)

                self.logger.info(
                    'Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}, cls_loss:{:.1f}, pod_spatial:{:.1f}, pod_flat:{:.1f}'
                    .format(epoch + 1, self.nb_epochs, str(i).zfill(3), str(len(train_loader)).zfill(3),
                            self.scheduler.get_last_lr()[0],
                            loss.item(), cls_loss.item(), pod_spatial.item(), pod_flat.item()))
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
        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, steps_per_epoch=len(train_loader),
                                                 epochs=self.nb_epochs,
                                                 pct_start=0.2)

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
            val_result, val_result2 = self.validate(low_range, high_range, val_loader_base, val_loader_seen, val_loader_new,
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
                self.logger.info('current other metrics:, mean_p_c:{}, mean_r_c:{}, mean_f_c:{}, precision_o:{}, recall_o:{}, f1_o:{}'
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
            for cls, score in enumerate(val_result['seen'][1]):
                real_class = coco_fake2real[cls]
                real_class = coco_ids_to_cats[real_class]
                self.writer.add_scalar("AP/{}".format(real_class), score, global_step=high_range)

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
                self.model_save_path, self.dataset_name, self.model_name,
                low_range, high_range))

        if self.kd_loss or self.pseudo_label:
            self.old_model = deepcopy(self.model).eval()

        # ----------
        # Sample and save protos from train dataset 
        # ----------

        if self.replay:
            start_time = time.time()
            # Sample protos on main process
            if self.rank == 0:
                train_dataset.transform = self.val_transforms  # sample protos without data augmentation

                # Random or Herding
                if self.sample_method == 'random':
                    sample_ds = random_sample_protos(train_dataset, low_range, high_range, self.num_protos)
                elif self.sample_method == 'herding':
                    loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                         num_workers=self.num_workers)
                    if self.args.fix_budget:
                          self.num_protos = int(np.ceil(self.args.buffer_size/self.num_classes))
                          sample_ds,cls_index = icarl_sample_protos_buffer(self.model_without_ddp, low_range, high_range,
                                                    train_dataset, loader, self.num_protos, self.image_id_set, self.logger)
                    else:
                           sample_ds = icarl_sample_protos(self.model_without_ddp, low_range, high_range,train_dataset, loader, self.num_protos
                                                        , self.image_id_set, self.logger)

                else:
                    raise ValueError(f"sample_method {self.sample_method} not supported !!!")

                # Concatenate protos to old dataset
                # todo append sample dataset to old_dataset list
                if self.args.fix_budget:
                    self.old_dataset.add(sample_ds,cls_index)
                else:
                    self.old_dataset.append(sample_ds)

                num_all_protos = 0
                if not self.args.fix_budget:
                    for dataset in self.old_dataset:
                       num_all_protos += len(dataset)
                    self.logger.info('Current stage sampled protos:{}, total old dataset length:{}'
                                     .format(len(sample_ds), num_all_protos))
                else:
                  self.logger.info('Current stage sampled protos:{}, total old dataset length:{}'
                                       .format(len(sample_ds), len(self.old_dataset.get_bufferset(self.args.buffer_size))))
                self.logger.info('Current Image ID set length {}'
                                 .format(len(self.image_id_set)))
                self.logger.info('Sample protos time:{}'.format(time.time() - start_time))

                # Save old dataset to disk
                torch.save(self.old_dataset, '{}/{}_{}_{}to{}_old_dataset.pth'.format(
                    self.model_save_path, self.dataset_name, self.model_name,
                    low_range, high_range))

                self.logger.info('Saved old dataset on rank 0, finished')
                train_dataset.transform = self.train_transforms  # resume the train transform for training

            # Load old dataset from disk for other process
            dist.barrier()
            if self.rank > 0:
                self.old_dataset = torch.load('{}/{}_{}_{}to{}_old_dataset.pth'.format(
                    self.model_save_path, self.dataset_name, self.model_name,
                    low_range, high_range))

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
                        torch.cat(preds_regular).cpu(), torch.cat(targets).cpu(), thre = 0.8)
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

                logits = output['logits']
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

        mAP_list = np.zeros((self.total_classes-self.base_classes) // self.task_size + 1)
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
            if self.args.fix_budget:
                self.old_dataset=buffer()

        for low_range, high_range in incremental_stages:
            # ----------
            # Getting Dataset and Dataloader
            # ----------

            # Build train dataset
            train_dataset_without_old = build_dataset(self.dataset_name, self.root_dir, low_range, high_range,
                                          phase='train', transform=self.train_transforms)
            self.logger.info('Current incremental stage:({},{}), dataset length:{}'
                             .format(low_range, high_range, len(train_dataset_without_old)))
            train_dataset = train_dataset_without_old

            # Concatenate protos with new datasets
            if self.replay and self.old_dataset and low_range!=0:
                if self.args.fix_budget:
                    train_dataset_with_old = torch.utils.data.ConcatDataset([train_dataset,self.old_dataset.get_bufferset(self.args.buffer_size)])
                else:
                    train_dataset_with_old = [train_dataset]
                    train_dataset_with_old.extend(self.old_dataset)
                    train_dataset_with_old = torch.utils.data.ConcatDataset(train_dataset_with_old)

                self.logger.info(
                    'Current incremental stage samples with old samples: {}\n'.format(len(train_dataset_with_old)))
                train_dataset = train_dataset_with_old

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
            mAP_meter.update(mAP)
            mAP_list[(high_range-self.base_classes)//self.task_size] = mAP
            
            self._after_task(low_range, high_range, train_dataset=train_dataset_without_old)
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
            expe_name = self.args.output_name
         #   git_hash = Repo(os.getcwd()).git.rev_parse("HEAD")
            params = f"LR:{self.lr}, epoch:{self.end_epoch}, BS:{self.batch_size}, GPU:{self.world_size}, Anchor:{self.num_protos}"
            print_to_excel(self.excel_path, expe_name, ds_name, self.base_classes, 
                self.task_size, self.total_classes, params, mAP_list, metrics, git_hash=None)
        self.logger.info('mAP list:{}'.format(mAP_list))
        # so that the logger won't output many times when tuning params
        del self.logger


