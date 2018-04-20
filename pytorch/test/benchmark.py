from __future__ import print_function
import argparse
from glob import glob
import os
import sys
import time
import torch
import nvvl
import psutil

from dataloading.dataloaders import get_loader

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True,
        help='folder of mp4/png files')
parser.add_argument('--sleep', type=float, required=True,
        help='dummy computation time')
parser.add_argument('--loader', type=str, required=True,
        help='dataloader: pytorch/NVVL/lintel')
parser.add_argument('--batchsize', type=int, default=8,
        help='batch size loaded')
parser.add_argument('--frames', type=int, default=3,
        help='number of frames in each loaded sequence')
parser.add_argument('--is_cropped', action='store_true',
        help='crop input frames?')
parser.add_argument('--crop_size', type=int, nargs='+', default=[-1, -1],
        help='[height, width] for input crop')
parser.add_argument('--fp16', action='store_true',
        help='load data in fp16?')

def main(args):

    assert args.sleep >= 0.0, print('Computation time must be >=0.0s')
    print(str(args) + '\n')

    loader, batches = get_loader(args)

    counter = 0
    data_time_sum = 0
    iter_time_sum = 0
    cpu_sum = 0
    mem_sum = 0

    for epoch in range(2):

        start = time.time()

        for i,x in enumerate(loader, 1):

            if args.loader != 'NVVL':
                x = x.cuda()
                if args.fp16:
                    x = x.half()

            if epoch > 0:
                counter += 1
                end = time.time()
                data_t = end-start
                if args.sleep > 0.0:
                    time.sleep(args.sleep)
                end = time.time()
                iter_t = end-start
                data_time_sum += data_t
                iter_time_sum += iter_t
                c = psutil.cpu_percent()
                cpu_sum += c
                m = psutil.virtual_memory().percent
                mem_sum += m
                start = time.time()

    data_time_ave = data_time_sum / counter
    iter_time_ave = iter_time_sum / counter
    cpu_ave = cpu_sum / counter
    mem_ave = mem_sum / counter
    print("Data loading time avg, iteration time avg, cpu load avg, memory usage avg")
    print("%.5f %.5f %.2f %.2f" % (data_time_ave, iter_time_ave, cpu_ave, mem_ave))

if __name__=='__main__':
    main(parser.parse_args())
