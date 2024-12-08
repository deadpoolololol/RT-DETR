"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    # training config
    cfg = YAMLConfig(
        args.config, # config file path
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )

    # create a training or validation solver
    solver = TASKS[cfg.yaml_cfg['task']](cfg) # task:detection
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default="configs/rtdetr/rtdetr_r50vd_6x_visdrone.yml",type=str, )
    # parser.add_argument('--resume', '-r',default=r"E:\Python\Project\2024.11.6 RT-DETR\rtdetr_pytorch\output\rtdetr_r50vd_6x_visdrone\20241208_120000\checkpoint0017.pth", type=str, )
    parser.add_argument('--resume', '-r',type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, help='seed',)
    args = parser.parse_args()

    main(args)
