import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_git_status, check_img_size, check_requirements, \
    check_file, check_yaml, check_suffix, print_args, print_mutation, set_logging, one_cycle, colorstr, methods
from utils.downloads import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_labels, plot_evolve
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.metrics import fitness
from utils.loggers import Loggers
from utils.callbacks import Callbacks

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

class AdaptiveBNEval(object):
    def __init__(self, model, opt, device, hyp) -> None:
        super().__init__()
        self.model = model
        self.opt = opt
        self.device = device
        self.hyp = hyp

        batch_size, rank = opt.batch_size, -1
        cuda = device.type != 'cpu'
        init_seeds(1 + rank)
        # with open(opt.data) as f:
        #     data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
        with torch_distributed_zero_first(rank):
            data_dict = check_dataset(opt.data)  # check
        self.data_dict = data_dict

        nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
        names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
        assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

        train_path = data_dict['train']
        test_path = data_dict['val']

        # Image sizes
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

        # Trainloader
        dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt.single_cls,
                                                hyp=hyp, augment=False, rank=rank, workers=opt.workers)
        mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
        nb = len(dataloader)  # number of batches
        assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

        # Process 0
        if rank in [-1, 0]:
            testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt.single_cls,  # testloader
                                        hyp=hyp, rank=-1, workers=opt.workers, pad=0.5)[0]

            labels = np.concatenate(dataset.labels, 0)

            # Anchors
            check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

        # Model parameters
        hyp['box'] *= 3. / nl  # scale to layers
        hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
        hyp['label_smoothing'] = 0.0
        model.nc = nc  # attach number of classes to model
        model.hyp = hyp  # attach hyperparameters to model
        model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
        model.names = names

        # Start training
        t0 = time.time()
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        scaler = amp.GradScaler(enabled=cuda)
        compute_loss = ComputeLoss(model)  # init loss class
        epoch = 0

        self.cuda = cuda
        self.batch_size = batch_size
        self.imgsz_test = imgsz_test
        self.testloader = testloader
        self.trainloader = dataloader
        self.device = device


    def __call__(self, compact_model):
        compact_model.train()
        with torch.no_grad():
            for i, (imgs, targets, paths, _) in tqdm(enumerate(self.trainloader), total=len(self.trainloader)):
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

                # Forward
                with amp.autocast(enabled=self.cuda):
                    pred = compact_model(imgs)  # forward
                    # loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                
                if i > 20:
                    break

        # (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
        results, _, _ = val.run(self.data_dict,
                                batch_size=self.batch_size * 2,
                                imgsz=self.imgsz_test,
                                conf_thres=0.001,
                                iou_thres=0.6,
                                model=compact_model,
                                single_cls=self.opt.single_cls,
                                dataloader=self.testloader,
                                save_json=False,
                                plots=False)

        return results[2]
