import datetime
import time
from MultiLabelIncremental_protoConflict import MultiLabelIncremental
from MultiLabelIncremental_KRT import MultiLabelIncremental_KRT
from MultiLabelIncremental_base import MultiLabelIncremental_Base
from MultiLabelIncremental_prs import MultiLabelIncremental_PRS
from MultiLabelIncremental_cuter import MultiLabelIncremental_CUTER
from map_eval import MultiLabelEval
from MultiLabelIncremental_er import MultiLabelIncremental_ER
from MultiLabelIncremental_ocdm import MultiLabelIncremental_OCDM
import torch.distributed as dist
import argparse
import yaml
import torch
import os
import  random
import torch.backends.cudnn as cudnn
import numpy as np

parser = argparse.ArgumentParser(description='MultilabelIncremental Training')
parser.add_argument('--local_rank', default=0, type=int, help='local rank for DistributedDataParallel')
parser.add_argument('--options', nargs='*')
parser.add_argument('--output_name', type=str)
parser.add_argument('--seed',default=3407 ,type=int)


def load_options(args, options):
    for o in options:
        with open(o) as f:
            config = yaml.safe_load(f)
            for k, v in config.items():
                if k not in args: # Config by terminal
                    setattr(args, k, v)


def main():
    args = parser.parse_args()
    if args.options:
        load_options(args, args.options)

    seed =args.seed
    print(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark=True

    # Distributed
    if 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
    print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
          % (args.rank, args.world_size))

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank, timeout=datetime.timedelta(seconds=5400))

    origin_output_name = args.output_name
    args.output_name = origin_output_name + f'_b{args.base_classes}i{args.task_size}' + f'_{args.num_protos}Proto'
    args.logger_dir = 'logs/' + args.output_name 
    args.tensorboard_dir = 'tensorboard/' + args.output_name 
    args.model_save_path = 'saved_models/' + args.output_name 
    
    if args.rank == 0:
        print("Arguments:")
        for k, v in sorted(vars(args).items()):
            print(k, '=', v)


    if args.arch == 'tresnet':
        multi_incremental = MultiLabelIncremental(args)
        multi_incremental.train()
    elif args.arch == 'ICA':
        multi_incremental = MultiLabelIncremental_KRT(args)
        multi_incremental.train()
    elif args.arch == 'resnet' or args.arch == 'vits' or args.arch == 'vitb':
        multi_incremental = MultiLabelIncremental_Base(args)
        multi_incremental.train()
    elif args.arch == 'resnet_ocdm' or args.arch == 'vits_ocdm' or args.arch == 'vitb_ocdm':
        multi_incremental = MultiLabelIncremental_OCDM(args)
        multi_incremental.train()
    elif args.arch == 'resnet_er' or args.arch == 'vits_er' or args.arch == 'vitb_er':
        multi_incremental = MultiLabelIncremental_ER(args)
        multi_incremental.train()
    elif args.arch == 'resnet_prs' or args.arch == 'vits_prs' or args.arch == 'vitb_prs':
        multi_incremental = MultiLabelIncremental_PRS(args)
        multi_incremental.train()
    elif args.arch == 'vits_cuter':
        multi_incremental = MultiLabelIncremental_CUTER(args)
        multi_incremental.train()
    elif args.arch == 'vits_eval':
        multi_incremental = MultiLabelEval(args)
        multi_incremental.train()
    else:
        print('error')
    
    del multi_incremental


if __name__ == '__main__':
    main()
