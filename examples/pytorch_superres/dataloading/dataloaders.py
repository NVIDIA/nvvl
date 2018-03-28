import sys
import copy
from glob import glob
import math
import os

import torch
from torch.utils.data import DataLoader

from dataloading.datasets import imageDataset
import nvvl

class NVVL():
    def __init__(self, frames, is_cropped, crop_size, root,
                 batchsize=1, device_id=0,
                 shuffle=False, distributed=False, fp16=False):

        self.root = root
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.distributed = distributed
        self.frames = frames
        self.device_id = device_id

        self.is_cropped = is_cropped
        self.crop_size = crop_size

        self.files = glob(os.path.join(self.root, '*.mp4'))

        if len(self.files) < 1:
            print(("[Error] No video files in %s" % (self.root)))
            raise LookupError

        if fp16:
            tensor_type = 'half'
        else:
            tensor_type = 'float'

        self.image_shape = nvvl.video_size_from_file(self.files[0])

        height = max(self.image_shape.height, self.crop_size[0])
        width = max(self.image_shape.width, self.crop_size[1])
        # Frames are enforced to be mod64 in each dimension
        # as required by FlowNetSD convolutions
        height = int(math.floor(height/64.)*64)
        width = int(math.floor(width/64.)*64)

        processing = {"input" : nvvl.ProcessDesc(type=tensor_type,
                                                 height=height,
                                                 width=width,
                                                 random_crop=self.is_cropped,
                                                 random_flip=False,
                                                 normalized=False,
                                                 color_space="RGB",
                                                 dimension_order="cfhw",
                                                 index_map=[0, 1, 2])}

        dataset = nvvl.VideoDataset(self.files,
                                    sequence_length=self.frames,
                                    device_id=self.device_id,
                                    processing=processing)

        self.loader = nvvl.VideoLoader(dataset,
                                  batch_size=self.batchsize,
                                  shuffle=self.shuffle,
                                  distributed=self.distributed)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)


def get_loader(args):

    if args.loader == 'pytorch':

        dataset = imageDataset(
            args.frames,
            args.is_cropped,
            args.crop_size,
            os.path.join(args.root, 'train'),
            args.batchsize,
            args.world_size)

        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        train_loader = DataLoader(
            dataset,
            batch_size=args.batchsize,
            shuffle=(sampler is None),
            num_workers=10,
            pin_memory=True,
            sampler=sampler,
            drop_last=True)

        effective_bsz = args.batchsize * float(args.world_size)
        train_batches = math.ceil(len(dataset) / float(effective_bsz))

        dataset = imageDataset(
            args.frames,
            args.is_cropped,
            args.crop_size,
            os.path.join(args.root, 'val'),
            args.batchsize,
            args.world_size)

        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        val_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            sampler=sampler,
            drop_last=True)

        val_batches = math.ceil(len(dataset) / float(args.world_size))

    elif args.loader == 'NVVL':

        train_loader = NVVL(
            args.frames,
            args.is_cropped,
            args.crop_size,
            os.path.join(args.root, 'train'),
            batchsize=args.batchsize,
            shuffle=True,
            distributed=True,
            device_id=args.rank % 8,
            fp16=args.fp16)

        train_batches = len(train_loader)

        val_loader = NVVL(
            args.frames,
            args.is_cropped,
            args.crop_size,
            os.path.join(args.root, 'val'),
            batchsize=1,
            shuffle=True,
            distributed=True,
            device_id=args.rank % 8,
            fp16=args.fp16)

        val_batches = len(val_loader)

        sampler = None

    else:

        raise ValueError('%s is not a valid option for --loader' % args.loader)

    return train_loader, train_batches, val_loader, val_batches, sampler
