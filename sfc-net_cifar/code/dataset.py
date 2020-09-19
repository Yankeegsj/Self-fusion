# Data Preprocess
import torchvision
import torch
import torchvision.transforms as transforms
import numpy as np
import random
def calc_dataset_stats(dataset, axis=0, ep=1e-7):
    return (np.mean(dataset, axis=axis) / 255.0).tolist(), (
            np.std(dataset + ep, axis=axis) / 255.0).tolist()

class cifar10():
    def __init__(self,padding=4,data_root=None,batch_size=128,num_workers=0,seed=1):

        self.CIFAR10_LABELS_LIST = [
                                'airplane',
                                'automobile',
                                'bird',
                                'cat',
                                'deer',
                                'dog',
                                'frog',
                                'horse',
                                'ship',
                                'truck'
                            ]
        '''
    if args.dataset=='cifar10':
        args.mean_rgb=[125.307, 122.950, 113.865]
        args.std_rgb=[62.993, 62.089, 66.705]
        args.test_batch_size=400 if args.batch_size<400 else args.batch_size #mxnet issues: test_batch>=train_batch
    elif args.dataset=='cifar100':
        args.num_classes=100
        args.mean_rgb=[129.304, 124.070, 112.434]
        args.std_rgb=[68.170, 65.392, 70.418]
        '''
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((125.307/255., 122.950/255., 113.865/255.), (1./62.993, 1./62.089, 1./66.705))
        ])
        transform_test  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((125.307/255., 122.950/255., 113.865/255.), (1./62.993, 1./62.089, 1./66.705))#读取数据减去均值，除以标准差，使得数据满足标准正态分布
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=data_root, transform=transform_train, train=True, download=False)
        test_dataset  = torchvision.datasets.CIFAR10(root=data_root, transform=transform_test, train=False, download=False)
        self.train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,worker_init_fn=np.random.seed(seed), shuffle=True,pin_memory=True)
        self.test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size=100, num_workers=num_workers,worker_init_fn=np.random.seed(seed), shuffle=False,pin_memory=True)


class cifar100():
    def __init__(self,padding=4,data_root=None,batch_size=128,num_workers=0,seed=1):
        '''
    if args.dataset=='cifar10':
        args.mean_rgb=[125.307, 122.950, 113.865]
        args.std_rgb=[62.993, 62.089, 66.705]
        args.test_batch_size=400 if args.batch_size<400 else args.batch_size #mxnet issues: test_batch>=train_batch
    elif args.dataset=='cifar100':
        args.num_classes=100
        args.mean_rgb=[129.304, 124.070, 112.434]
        args.std_rgb=[68.170, 65.392, 70.418]
        '''
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((129.304/255., 124.070/255., 112.434/255.), (1./68.170, 1./65.392, 1./70.418))
        ])
        transform_test  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((129.304/255., 124.070/255., 112.434/255.), (1./68.170, 1./65.392, 1./70.418))#读取数据减去均值，除以标准差，使得数据满足标准正态分布
        ])
        train_dataset = torchvision.datasets.CIFAR100(root=data_root, transform=transform_train, train=True, download=False)
        test_dataset  = torchvision.datasets.CIFAR100(root=data_root, transform=transform_test, train=False, download=False)
        self.train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,worker_init_fn=np.random.seed(seed), shuffle=True,pin_memory=True)
        self.test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size=100, num_workers=num_workers,worker_init_fn=np.random.seed(seed), shuffle=False,pin_memory=True)

