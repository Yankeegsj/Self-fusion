#coding=gbk
import os
import sys
sys.path.append(os.path.abspath((__file__)))
from options import global_variables as opt_conf
from tools.progress.bar import Bar
from tools.terminal_log import create_log_file_terminal,save_opt,create_exp_dir
from tools.draw_acc_map import draw_acc_loss_line
import tools.godblessdbg as godblessdbg
from torch.utils import data
from tools.log import AverageMeter
from tools.progress.bar import Bar
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import time
import numpy as np
import random
from dataset import cifar10,cifar100
import utils
import glob
import math
def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    torch.manual_seed(seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True

def _initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

def validate(model, testloader, criterion,opt):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    bar = Bar('Validate', max=len(testloader))
    for step, (inputs, labels) in enumerate(testloader):
        inp = torch.autograd.Variable(inputs.cuda())
        target = torch.autograd.Variable(labels.cuda())
        with torch.no_grad():#ÔÚÑéÖ¤Ê±¼ÓÉÏÕâ¾äÄÜ´ó´ó¼õÉÙÏÔ´æ
            output = model(inp)
        loss = criterion(output, target)
        losses.update(loss.item(), labels.size(0))
        prec1,prec5= utils.accuracy(output, target, topk=(1,5))
        top1.update(prec1.data, labels.size(0))
        top5.update(prec5.data, labels.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Top1 :{top1:.4f} | Top5 :{top5:.4f} | Loss:{loss:.4f} '.format(
            batch=step + 1,
            size=len(testloader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            top1=top1.avg,
            top5=top5.avg,
            loss=losses.avg
        )
        bar.next()
    bar.finish()
    return top1.avg, top5.avg,losses.avg



if __name__ == '__main__':
    
    option = dict([arg.split('=') for arg in sys.argv[1:]])
    opt=opt_conf(**option)
    seed_torch(opt.seed)
    log_output=create_log_file_terminal(opt.log_txt_path)
    save_opt(opt,opt.log_txt_path)
    create_exp_dir(opt.log_root_path,glob.glob('./code/*.py'))
    log_output.info(os.path.realpath(__file__))
    if not torch.cuda.is_available():
        log_output.info('no gpu device available')
        sys.exit(1)

    if opt.dataset=='cifar10':
        dataset_loader=cifar10(padding=opt.train_padding,data_root=opt.data_root,
                                batch_size=opt.batch_size,num_workers=opt.num_workers,seed=opt.seed)
    elif opt.dataset=='cifar100':
        dataset_loader=cifar100(padding=opt.train_padding,data_root=opt.data_root,
                                batch_size=opt.batch_size,num_workers=opt.num_workers,seed=opt.seed)
    from importlib import import_module
    net=import_module('models.{}'.format(opt.model))
    exec('model=net.{}(num_classes=opt.num_classes)'.format(opt.model))
    
    model.apply(_initialize_weights)
    model = model.cuda()
    log_output.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer=torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=opt.lr_init,
                                momentum=opt.momentum,weight_decay=opt.weight_decay,nesterov=True)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=opt.decay_gamma)

    train_loss_list=[]
    train_acc_list=[]
    test_loss_list=[]
    test_acc_list=[]
    accuracy_top1=0.
    for epoch in range(1,opt.num_epochs+1):
        if isinstance(opt.decay_epoch,list) and epoch in [1]+opt.decay_epoch:
            scheduler.step()
        elif epoch==1 or (isinstance(opt.decay_epoch,int) and epoch%opt.decay_epoch==0):
            scheduler.step()
        lr = scheduler.get_lr()[0]
        log_output.info('epoch {} lr {}'.format(epoch,lr))

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acces = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        model.train()
        bar = Bar('Training', max=len(dataset_loader.train_loader))
        flip_count=0
        for step, (inputs, labels) in enumerate(dataset_loader.train_loader):
            data_time.update(time.time() - end)
            inp = torch.autograd.Variable(inputs.cuda())
            target = torch.autograd.Variable(labels.cuda())
            output = model(inp)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), labels.size(0))
            prec1,prec5= utils.accuracy(output, target, topk=(1,5))
            top1.update(prec1.data, labels.size(0))
            top5.update(prec5.data, labels.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Top1 :{top1:.3f} | Top5 :{top5:.3f} | Loss:{loss:.4f}'.format(
                batch=step + 1,
                size=len(dataset_loader.train_loader),
                data=data_time.val,
                bt=batch_time.val,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                top1=top1.avg,
                top5=top5.avg,
                loss=losses.avg,
            )
            bar.next()

        bar.finish()
        log_output.info('train_acc1:{} train_acc5:{} train_loss:{} '.format(top1.avg,top5.avg,losses.avg))
        
        train_loss_list.append(losses.avg)
        train_acc_list.append(top1.avg)

        loss_pic_save_path=os.path.join(opt.log_root_path,'train_loss.pdf')
        acc_pic_save_path=os.path.join(opt.log_root_path,'train_acc.pdf')
        draw_acc_loss_line(train_loss_list, train_acc_list, loss_pic_save_path, acc_pic_save_path, 'train')

        current_top1_accuracy,current_top5_accuracy,test_loss= validate(model, dataset_loader.test_loader, criterion,opt)
        log_output.info('test_acc1:{} test_acc5:{} test_loss:{} '.format(current_top1_accuracy,current_top5_accuracy,test_loss))
        test_loss_list.append(test_loss)
        test_acc_list.append(current_top1_accuracy)
        loss_pic_save_path=os.path.join(opt.log_root_path,'test_loss.pdf')
        acc_pic_save_path=os.path.join(opt.log_root_path,'test_acc.pdf')
        draw_acc_loss_line(test_loss_list, test_acc_list, loss_pic_save_path, acc_pic_save_path, 'test')


        if accuracy_top1 < current_top1_accuracy:
            accuracy_top1 = current_top1_accuracy
            respond_epoch=epoch
            state = {
            'net': model.state_dict() ,
            'acc': accuracy_top1,
            'epoch': respond_epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, os.path.join(opt.log_root_path,'best.pth'))
        state = {
        'net': model.state_dict() ,
        'acc': current_top1_accuracy,
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(opt.log_root_path,'latest.pth'))

    log_output.info('Train complete. Accuracy :{0} \t epoch:{1}'.format(accuracy_top1,  respond_epoch))
    log_output.info(godblessdbg.end)
    log_output.info('log location {0}'.format(opt.log_root_path))