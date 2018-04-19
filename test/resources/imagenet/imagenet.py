import os
import shutil
import time
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

best_prec1 = 0


def _load_hyperparameters(hyperparameters):
    logger.info("Load hyperparameters")
    # number of data loading workers (default: 4)
    workers = hyperparameters.get('workers', 4)
    # number of total epochs to run (default: 90)
    epochs = hyperparameters.get('epochs', 90)
    # manual epoch number (useful on restarts) (default: 0)
    start_epoch = hyperparameters.get('start_epoch', 0)
    # mini_batch size (default: 256)
    batch_size = hyperparameters.get('batch_size', 256)
    # initial learning rate (default: 0.1)
    lr = hyperparameters.get('lr', 0.1)
    # momentum (default: 0.9)
    momentum = hyperparameters.get('momentum', 0.9)
    # weight decay (default: 1e-4)
    weight_decay = hyperparameters.get('weight_decay', 1e-4)
    # print frequency (default: 10)
    print_freq = hyperparameters.get('print_freq', 10)
    # path to latest checkpoint (default: none)
    resume = hyperparameters.get('resume', '')
    # evaluate model on validation set
    evaluate = hyperparameters.get('evaluate', True)
    # number of distributed processes
    world_size = hyperparameters.get('world_size', 1)
    # url used to set up distributed training
    dist_url = hyperparameters.get('dist_url', 'tcp://224.66.41.62:23456')
    # distributed backend
    backend = hyperparameters.get('dist_backend', 'gloo')

    logger.info(
        'workers: {}, epochs: {}, start_epoch: {}, '.format(workers, epochs, start_epoch) +
        'batch_size: {}, lr: {}, momentum: {}, '.format(batch_size, lr, momentum) +
        'weight_decay: {}, print_freq: {}, resume: {}, '.format(weight_decay, print_freq, resume) +
        'evaluate: {}, world_size: {}, dist_url: {}, backend: {}'.format(evaluate, world_size,
                                                                         dist_url, backend)
    )
    return workers, epochs, start_epoch, batch_size, lr, momentum, weight_decay, print_freq, \
           evaluate, resume, world_size, dist_url, backend


def _get_train_data_loader(batch_size, training_dir, train_sampler, workers):
    logger.info("Get train data loader")
    return torch.utils.data.DataLoader(
        training_dir, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)


def _get_val_data_loader(batch_size, val_dir, normalize, workers):
    logger.info("Get val data loader")
    return torch.utils.data.DataLoader(
        datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)


def _validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = _accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def _save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def _adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(channel_input_dirs, num_gpus, hosts, host_rank, master_addr, master_port,
          hyperparameters):
    training_dir = channel_input_dirs['training']
    val_dir = channel_input_dirs['validation']

    workers, epochs, start_epoch, batch_size, lr, momentum, weight_decay, print_freq, resume, \
    evaluate, world_size, dist_url, backend = _load_hyperparameters(hyperparameters)

    is_distributed = len(hosts) > 1 and backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    cuda = num_gpus > 0
    logger.debug("Number of gpus available - {}".format(num_gpus))
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    if is_distributed:
        # Initialize the distributed environment.
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L69
        world_size = len(hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        dist.init_process_group(backend=backend, rank=host_rank, world_size=world_size)
        logger.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            backend,
            dist.get_world_size()) + 'Current host rank is {}. Using cuda: {}. Number of gpus: {}'.format(
            dist.get_rank(), torch.cuda.is_available(), num_gpus))

    # create model
    model = models.resnet50()

    if is_distributed and cuda:
        # multi-machine multi-gpu case
        logger.debug("Multi-machine multi-gpu: using DistributedDataParallel.")
        model = torch.nn.parallel.DistributedDataParallel(model.cuda())
    elif cuda:
        # single-machine multi-gpu case
        logger.debug("Single-machine multi-gpu: using DataParallel().cuda().")
        model = torch.nn.DataParallel(model.cuda()).cuda()
    else:
        # single-machine or multi-machine cpu case
        logger.debug("Single-machine/multi-machine cpu: using DataParallel.")
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        training_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = _get_train_data_loader(batch_size, training_dir, train_sampler, workers)
    val_loader = _get_val_data_loader(batch_size, val_dir, normalize, workers)

    if evaluate:
        _validate(val_loader, model, criterion, print_freq)
        return

    for epochs in range(start_epoch, epochs):
        if is_distributed:
            train_sampler.set_epoch(epochs)
        _adjust_learning_rate(optimizer, epochs)

        # train for one epoch
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.train()

        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = _accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epochs, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        # evaluate on validation set
        prec1 = _validate(val_loader, model, criterion, print_freq)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        _save_checkpoint({
            'epoch': epochs + 1,
            'arch': "resnet50",
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def test(model, test_loader, cuda):
    return


def model_fn(model_dir):
    return
