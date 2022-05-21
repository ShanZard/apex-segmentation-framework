from datetime import datetime
import os
import os.path as osp
import numpy as np 
import random
# PyTorch includes
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import yaml
from train_process import Trainer
import torch.nn 
import torch.backends.cudnn as cudnn
# Custom includes
from dataloaders import dataloader as DL
from dataloaders import custom_transforms as tr
from models import create_model
from net import convnext_tiny
from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel as DDP
from unet import Unet
here = osp.dirname(osp.abspath(__file__))
### CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 train.py
def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('--local_rank', type=int, default=0, help='gpu id')
    parser.add_argument('--resume', default=None, help='checkpoint path')
    parser.add_argument(
        '--datasetdir', type=str, default='/root/root/DAdataset/dadataset', help='test folder id contain images ROIs to test'
    )
    parser.add_argument(
        '--batch-size', type=int, default=1, help='batch size for training the model'
    )
    parser.add_argument(
        '--group-num', type=int, default=1, help='group number for group normalization'
    )
    parser.add_argument(
        '--max-epoch', type=int, default=400, help='max epoch'
    )
    parser.add_argument(
        '--stop-epoch', type=int, default=400, help='stop epoch'
    )
    parser.add_argument(
        '--interval-validate', type=int, default=1, help='interval epoch number to valide the model'
    )
    parser.add_argument(
        '--lr-model', type=float, default=2e-4, help='learning rate'
    )
    parser.add_argument(
        '--seed',type=int,default=26,help='set random seed'
    )
    parser.add_argument(
        '--lr-decrease-rate', type=float, default=0.95, help='ratio multiplied to initial lr',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    parser.add_argument(
        '--warmup_epoch',type=int,default=-1,help='warmup_epoch'
    )

    args = parser.parse_args()


    now = datetime.now()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)



    cuda = torch.cuda.is_available()
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
    torch.cuda.set_device(args.local_rank)  # 必须写！，还必须在下一句的前面，
	#torch.utils.launch也需要set_device， 所以必须写
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


    # 1. dataset
    composed_transforms_tr = transforms.Compose([
        tr.RandomScaleCrop(512),
        tr.RandomRotate(),
        tr.RandomFlip(),
        tr.elastic_transform(),
        tr.add_salt_pepper_noise(),
        tr.adjust_light(),
        tr.eraser(),
        tr.Normalize_tf(),        
        tr.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        tr.RandomScaleCrop(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    mydataset = DL.Segmentation(base_dir=args.datasetdir, split='test',
                                                         transform=composed_transforms_tr)
    train_sampler  = torch.utils.data.distributed.DistributedSampler(mydataset)                                                         
    mydataloader = DataLoader(mydataset, batch_size=args.batch_size, shuffle=False, num_workers=18, pin_memory=cuda,sampler=train_sampler)

    mydataset_val = DL.Segmentation( base_dir=args.datasetdir ,split='test',
                                       transform=composed_transforms_ts)
    val_sampler  = torch.utils.data.distributed.DistributedSampler(mydataset_val)                                       
    mydataloader_val = DataLoader(mydataset_val, batch_size=args.batch_size, shuffle=False, num_workers=18, pin_memory=cuda,sampler=val_sampler)

    # 2. model
    #model = create_model('DeepLabV3Plus',encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet', encoder_output_stride=16, decoder_channels=256, decoder_atrous_rates=(12, 24, 36), in_channels=3, classes=2, activation=None, upsampling=4, aux_params=None)
    model=convnext_tiny(using_amp=True)
    # model=Unet(using_amp=True)
    start_epoch = 0
    start_iteration = 0

    # 3. optimizer


    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        start_epoch = checkpoint['epoch'] + 1
        start_iteration = checkpoint['iteration'] + 1
        optim_model.load_state_dict(checkpoint['optim_state_dict'])

    trainer = Trainer.Trainer(
        local_rank=args.local_rank,
        model=model,
        lr_gen=args.lr_model,
        lr_decrease_rate=args.lr_decrease_rate,
        loader=mydataloader,
        val_loader=mydataloader_val,
        out=args.out,
        max_epoch=args.max_epoch,
        stop_epoch=args.stop_epoch,
        interval_validate=args.interval_validate,
        batch_size=args.batch_size,
        warmup_epoch=args.warmup_epoch
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()
    

if __name__ == '__main__':
    main()
