#!/usr/bin/env python
# encoding: utf-8
# File Name: train_graph_moco.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/13 16:44
# TODO:

import os
import torch
import dgl.model_zoo.chem as zoo
import time
from util import adjust_learning_rate, AverageMeter
from NCE.NCEAverage import MemoryMoCo
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss
from graph_dataset import GraphDataset, batcher

import tensorboard_logger as tb_logger

try:
    from apex import amp, optimizers
except ImportError:
    pass

def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1-m, p1.detach().data)


def train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, opt):
    """
    one epoch training for moco
    """

    model.train()
    model_ema.eval()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()
    model_ema.apply(set_bn_train)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        graph_q, graph_k = batch.graph_q, batch.graph_q
        bsz = graph_q.batch_size

        # ===================forward=====================

        feat_q = model(graph_q)
        with torch.no_grad():
            feat_k = model_ema(graph_k)

        out = contrast(feat_q, feat_k)

        loss = criterion(out)
        prob = out[:, 0].mean()

        # ===================backward=====================
        optimizer.zero_grad()
        if opt.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)

        moment_update(model, model_ema, opt.alpha)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'prob {prob.val:.3f} ({prob.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_meter, prob=prob_meter))
            print(out.shape)

    return loss_meter.avg, prob_meter.avg


def main():

    args = parse_option()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # set the data loader
    data_folder = os.path.join(args.data_folder, 'train')


    train_dataset = GraphDataset(
            random_walk_hops=args.random_walk_hops,
            subgraph_size=args.subgraph_size,
            restart_prob=args.restart_prob,
            hidden_size=args.hidden_size
            )
    print(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=20,
            collate_fn=batcher(),
            shuffle=True,
            num_workers=args.num_workers)

    # create model and optimizer
    n_data = len(train_dataset)

    if args.model == 'gcn':
        model = zoo.GCNClassifier()
        model_ema = zoo.GCNClassifier()
    elif args.model == 'gat':
        model = zoo.GATClassifier()
        model_ema = zoo.GATClassifier()
    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

    # copy weights from `model' to `model_ema'
    moment_update(model, model_ema, 0)

    # set the contrast memory and criterion
    contrast = MemoryMoCo(128, n_data, args.nce_k, args.nce_t, args.softmax).cuda(args.gpu)

    criterion = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion = criterion.cuda(args.gpu)

    model = model.cuda()
    model_ema = model_ema.cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            if args.moco:
                model_ema.load_state_dict(checkpoint['model_ema'])

            if args.amp and checkpoint['opt'].amp:
                print('==> resuming amp state_dict')
                amp.load_state_dict(checkpoint['amp'])

            print("=> loaded successfully '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        loss, prob = train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('ins_loss', loss, epoch)
        logger.log_value('ins_prob', prob, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if args.moco:
                state['model_ema'] = model_ema.state_dict()
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state

        # saving the model
        print('==> Saving...')
        state = {
            'opt': args,
            'model': model.state_dict(),
            'contrast': contrast.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        if args.moco:
            state['model_ema'] = model_ema.state_dict()
        if args.amp:
            state['amp'] = amp.state_dict()
        save_file = os.path.join(args.model_folder, 'current.pth')
        torch.save(state, save_file)
        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
        # help release GPU memory
        del state
        torch.cuda.empty_cache()



if __name__ == '__main__':
    main()
