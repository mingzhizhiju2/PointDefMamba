import datetime
import os
import sys
import random
import time
import torch
import argparse
import numpy as np
from pathlib import Path
from tensorboardX import SummaryWriter

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import misc, dist_utils
from utils.logger import *
from utils.config import *

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser('PointDeformableMamba Model')
    parser.add_argument('--config', type=str, help='yaml config file')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # seed
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    
    # bn
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    
    # some args
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ids')
    parser.add_argument('--log_dir', type=str, default='scanobjectnn_objectbg', help='log directory')
    parser.add_argument('--val_freq', type=int, default=1, help='test freq')
    parser.add_argument('--vote', action='store_true', default=False, help='vote acc')
    parser.add_argument('--resume', action='store_true', default=False, help='autoresume training (interrupted by accident)')
    parser.add_argument('--test', action='store_true', default=False, help='test mode for certain ckpt')
    parser.add_argument('--finetune_model', action='store_true', default=True, help='finetune modelnet with pretrained weight')
    parser.add_argument('--scratch_model', action='store_true', default=False, help='training modelnet from scratch')
    
    # checkpoint
    parser.add_argument('--ckpts', type=str, default=None, help='test used ckpt path')
    parser.add_argument('--start_ckpts', type=str, default=None, help='reload used ckpt path')
    
    return parser.parse_args()

def main():
    # args
    args = get_args()
    seed_torch(args.seed)
    
    # CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.use_gpu = torch.cuda.is_available()

    # 初始化分布式环境
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    
    # 创建实验目录
    args.experiment_path = os.path.join(
        './experiments', 
        Path(args.config).stem, 
        Path(args.config).parent.stem,
        args.exp_name,
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    )
    args.tfboard_path = os.path.join(
        './experiments', 
        Path(args.config).stem, 
        Path(args.config).parent.stem, 
        'TFBoard',
        args.exp_name,
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    )
    args.log_name = Path(args.config).stem
    
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print(f'Create experiment path: {args.experiment_path}')
    
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print(f'Create TFBoard path: {args.tfboard_path}')
    
    # 日志设置
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    
    # TensorBoard
    if not args.test and args.local_rank == 0:
        train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
        val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
    else:
        train_writer = None
        val_writer = None
    
    # 配置
    config = get_config(args, logger=logger)
    
    # 批处理大小设置
    if args.distributed:
        assert config.total_bs % args.world_size == 0
        if config.dataset.get('train'):
            config.dataset.train.others.bs = config.total_bs // args.world_size
        if config.dataset.get('val'):
            config.dataset.val.others.bs = config.total_bs // args.world_size
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // args.world_size
    else:
        if config.dataset.get('train'):
            config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('val'):
            config.dataset.val.others.bs = config.total_bs
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs
    
    # 记录参数
    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)
    logger.info(f'Distributed training: {args.distributed}')
    
    # 设置随机种子
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed, deterministic=args.deterministic)
    
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()
    
    # 导入所需模块
    from tools import finetune_run_net as finetune
    from tools import test_run_net as test_net
    
    # 运行训练或测试
    if args.test:
        test_net(args, config)
    else:
        if args.finetune_model or args.scratch_model:
            finetune(args, config, train_writer, val_writer)
        else:
            logger.info("使用默认的finetune模式进行训练")
            finetune(args, config, train_writer, val_writer)

if __name__ == '__main__':
    main() 