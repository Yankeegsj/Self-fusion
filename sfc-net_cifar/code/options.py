import time
import os
import torch
import sys

class global_variables(object):
    model=''
    time_=time.localtime()
    time_=time.strftime('%Y-%m-%d-%H-%M',time_)

    dataset='cifar10'
    seed=1
    train_padding=4
    data_root='./cifar-10'

    log_root_path=''
    log_txt_path=''
    
    num_epochs   = 300
    #dataset
    num_workers=6
    num_classes=10
    batch_size   = 128

    #optimizer
    
    lr_init      = 0.1
    weight_decay=1e-4
    momentum=0.9
    decay_epoch  =[150,225]
    decay_gamma  =0.1
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s " % k)
                self.__dict__.update({k: v})
            tp = eval('type(self.{0})'.format(k))
            if tp == type(''):
                setattr(self, k, tp(v))
            else:
                setattr(self, k, eval(v))

        self.log_root_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'log/{}_{}_{}'.format(self.time_,self.model,self.dataset))

        self.log_txt_path=os.path.join(os.path.dirname(self.log_root_path),'{}_{}_{}.txt'.format(self.time_,self.model,self.dataset))
        if not os.path.exists(self.log_root_path):
            os.makedirs(self.log_root_path)

