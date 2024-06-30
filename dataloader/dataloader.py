import cv2
import torch
import numpy as np
from torch.utils import data
import random
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize

def random_mirror(A, B, gt):
    if random.random() >= 0.5:
        A = cv2.flip(A, 1)
        B = cv2.flip(B, 1)
        gt = cv2.flip(gt, 1)

    return A, B, gt

def random_scale(A, B, gt, scales):
    scale = random.choice(scales)
    sh = int(A.shape[0] * scale)
    sw = int(A.shape[1] * scale)
    A = cv2.resize(A, (sw, sh), interpolation=cv2.INTER_LINEAR)
    B = cv2.resize(B, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return A, B, gt, scale

class TrainPre(object):
    def __init__(self, norm_mean, norm_std, config):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.config = config

    def __call__(self, A, B, gt):
        A, B, gt = random_mirror(A, B, gt)
        if self.config.train_scale_array is not None:
            A, B, gt, scale = random_scale(A, B, gt, self.config.train_scale_array)

        A = normalize(A, self.norm_mean, self.norm_std)
        B = normalize(B, self.norm_mean, self.norm_std)
        gt = (gt>124)+0

        # crop_size = (self.config.image_height, self.config.image_width)
        # crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        # p_A, _ = random_crop_pad_to_shape(A, crop_pos, crop_size, 0)
        # p_B, _ = random_crop_pad_to_shape(B, crop_pos, crop_size, 0)
        # p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_gt = gt

        # p_A = p_A.transpose(2, 0, 1)
        p_A = A.transpose(2, 0, 1)

        # p_B = p_B.transpose(2, 0, 1)
        p_B = B.transpose(2, 0, 1)
        
        return p_A, p_B, p_gt

class ValPre(object):
    def __call__(self, A, B, gt):
        gt = (gt > 124)+0
        return A, B, gt

def get_train_loader(engine, dataset, config):
    data_setting = {'root': config.root_folder,
                    'A_format': config.A_format,
                    'B_format': config.B_format,
                    'gt_format': config.gt_format,
                    'class_names': config.class_names}

    train_preprocess = TrainPre(config.norm_mean, config.norm_std, config)

    # train_dataset = dataset(data_setting, "train", train_preprocess, config.batch_size * config.niters_per_epoch)
    train_dataset = dataset(data_setting, "train", train_preprocess)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler