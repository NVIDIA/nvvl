import pickle
import argparse
import logging as log
import os
import time

from math import ceil, floor
from tensorboardX import SummaryWriter

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data.distributed
from torch.multiprocessing import Process
from torch.autograd import Variable

from dataloading.dataloaders import get_loader

from model.model import VSRNet
from model.clr import cyclic_learning_rate

from apex import amp
try:
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')
parser.add_argument('--root', type=str, default='.',
                    help='input data root folder')
parser.add_argument('--frames', type=int, default = 3,
                    help='num frames in input sequence')
parser.add_argument('--is_cropped', action='store_true',
                    help='crop input frames?')
parser.add_argument('--crop_size', type=int, nargs='+', default=[256, 256],
                    help='[height, width] for input crop')
parser.add_argument('--batchsize', type=int, default=1,
                    help='per rank batch size')
parser.add_argument('--loader', type=str, default='NVVL',
                    help='dataloader: pytorch or NVVL')
parser.add_argument('--rank', type=int, default=0,
                    help='used for multi-process training. Can either be ' +
                    'or automatically set by uing \'python -m multiproc\'.')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of GPUS to use. Can either be manually set ' +
                    'or automatically set by using \'python -m multiproc\'.')
parser.add_argument('--dist-url', default='tcp://localhost:3567', type=str,
                    help='url for distributed init.')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--max_iter', type=int, default=1000,
                    help='num training iters')
parser.add_argument('--checkpoint_dir', type=str, default='.',
                    help='where to save checkpoints')
parser.add_argument('--job_name', type=str, default='default_job',
                    help='name for checkpoint folder')
parser.add_argument('--min_lr', type=float, default=0.000001,
                    help='min learning rate for cyclic learning rate')
parser.add_argument('--max_lr', type=float, default=0.00001,
                    help='max learning rate for cyclic learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0004,
                    help='ADAM weight decay')
parser.add_argument('--flownet_path', type=str,
                    default='flownet2-pytorch/networks/FlowNet2-SD_checkpoint.pth.tar',
                    help='FlowNetSD weights path')
parser.add_argument('--image_freq', type=int, default=100,
                    help='num iterations between image dumps to Tensorboard ')
parser.add_argument('--timing', action='store_true',
                    help="Time data loading and model training (default: False)")
parser.add_argument('--amp', action='store_true',
                    help="Enable amp for automatic mixed precision training")

def backward_inf_nan_hook(model):
    for name, mod in model.named_modules():
        #if len(list(mod.children())) == 0:
        mod.register_backward_hook(hook_gen(name))

def _maybe_iterable(x):
    try:
        iter(x)
        return x
    except TypeError:
        return [x]

def hook_gen(name):
    def hook(module, grad_input, grad_output):
        for x in _maybe_iterable(grad_output):
            if x is None:
                continue
            if torch.isnan(x).any():
                print('[OUT]{}({}): NAN'.format(name, type(module)))
            if (x.float().abs() == float('inf')).any():
                print('[OUT]{}({}): INF'.format(name, type(module)))
        for x in _maybe_iterable(grad_input):
            if x is None:
                continue
            if torch.isnan(x).any():
                print('[IN]{}({}): NAN'.format(name, type(module)))
            if (x.float().abs() == float('inf')).any():
                print('[IN]{}({}): INF'.format(name, type(module)))
    return hook

def main(args):

    if args.rank == 0:
        log.basicConfig(level=log.INFO)
        import random
        writer = SummaryWriter('/raid/jbarker/runs/' + str(random.randint(0, 100000)))
        writer.add_text('config', str(args))
    else:
        log.basicConfig(level=log.WARNING)
        writer = None

    torch.cuda.set_device(args.rank % torch.cuda.device_count())
    torch.manual_seed(args.seed + args.rank)
    torch.cuda.manual_seed(args.seed + args.rank)
    torch.backends.cudnn.benchmark = True

    log.info('Initializing process group')
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank)
    log.info('Process group initialized')

    log.info("Initializing dataloader...")
    train_loader, train_batches, val_loader, val_batches, sampler = get_loader(args)
    samples_per_epoch = train_batches * args.batchsize
    log.info('Dataloader initialized')

    if args.amp:
        amp_handle = amp.init(enabled=args.amp)
    model = VSRNet(args.frames, args.flownet_path, args.amp)
    model.cuda()
    model.train()
    for param in model.FlowNetSD_network.parameters():
        param.requires_grad = False

    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(model_params, lr=1, weight_decay=args.weight_decay)
    #optimizer = optim.SGD(model_params, lr=1,
    #                      momentum=0.99, weight_decay=args.weight_decay)
    stepsize = 2 * train_batches
    clr_lambda = cyclic_learning_rate(args.min_lr, args.max_lr, stepsize)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[clr_lambda])

    model = DDP(model, shared_param=True)

    # BEGIN TRAINING
    total_iter = 0
    while total_iter * args.world_size < args.max_iter:

        epoch = floor(total_iter / train_batches)
        if args.loader == 'pytorch':
            sampler.set_epoch(epoch)

        model.train()
        backward_inf_nan_hook(model)
        total_epoch_loss = 0.0

        sample_timer = 0.0
        data_timer = 0.0
        compute_timer = 0.0

        iter_start = time.perf_counter()

        # TRAINING EPOCH LOOP
        for i, inputs in enumerate(train_loader):

            if args.loader == 'NVVL':
                inputs = inputs['input']
            else:
                inputs = inputs.cuda(non_blocking=True)

            if args.timing:
                torch.cuda.synchronize()
                data_end = time.perf_counter()

            optimizer.zero_grad()

            im_out = total_iter % args.image_freq == 0
            loss = model(Variable(inputs), i, writer, im_out)

            total_epoch_loss += loss.item()

            if args.amp:
                with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    # after backward()
            else:
                loss.backward()

            optimizer.step()
            scheduler.step()

            if args.rank == 0:
                if args.timing:
                    torch.cuda.synchronize()
                    iter_end = time.perf_counter()
                    sample_timer += (iter_end - iter_start)
                    data_timer += (data_end - iter_start)
                    compute_timer += (iter_end - data_end)
                    torch.cuda.synchronize()
                    iter_start = time.perf_counter()
                writer.add_scalar('learning_rate', scheduler.get_lr()[0], total_iter)
                writer.add_scalar('train_loss', loss.item(), total_iter)
                if args.amp:
                    writer.add_scalar('loss_scale', scaled_loss.float()/loss.float(), total_iter)
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    if torch.isnan(param.grad).any() == 1 or (param.grad == float('inf')).any():
                        f = open('/raid/jbarker/checkpoints/grads/fp16_' + str(total_iter) + '_' + name + '.grad', 'wb')
                        pickle.dump(param.grad, f)
                    # things to look at:
                    #print(param.grad.norm())
                    #print(torch.isnan(param.grad).any())
                    #print(torch.max(param.grad.abs()))


            log.info('Rank %d, Epoch %d, Iteration %d of %d, loss %.5f' %
                    (dist.get_rank(), epoch, i+1, train_batches, loss.item()))

            total_iter += 1

        if args.rank == 0:
            if args.timing:
                sample_timer_avg = sample_timer / samples_per_epoch
                writer.add_scalar('sample_time', sample_timer_avg, total_iter)
                data_timer_avg = data_timer / samples_per_epoch
                writer.add_scalar('sample_data_time', data_timer_avg, total_iter)
                compute_timer_avg = compute_timer / samples_per_epoch
                writer.add_scalar('sample_compute_time', compute_timer_avg, total_iter)
            epoch_loss_avg = total_epoch_loss / train_batches
            log.info('Rank %d, epoch %d: %.5f' % (dist.get_rank(), epoch, epoch_loss_avg))

            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, args.job_name, 'weights_' + str(epoch) + '.pth'))


        model.eval()
        total_loss = 0
        total_psnr = 0

        for i, inputs in enumerate(val_loader):

            if args.loader == 'NVVL':
                inputs = inputs['input']
            else:
                inputs = inputs.cuda(non_blocking=True)

            log.info('Validation it %d of %d' % (i + 1, val_batches))
            loss, psnr = model(Variable(inputs), i, None)
            total_loss += loss.item()
            total_psnr += psnr.item()

        loss = total_loss / i
        psnr = total_psnr / i

        if args.rank == 0:
            writer.add_scalar('val_loss', loss, total_iter)
            writer.add_scalar('val_psnr', psnr, total_iter)
        log.info('Rank %d validation loss %.5f' % (dist.get_rank(), loss))
        log.info('Rank %d validation psnr %.5f' % (dist.get_rank(), psnr))

if __name__=='__main__':
    main(parser.parse_args())

