import torch.nn as nn
import torch
import math

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size = 1, stride= 1, padding= 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
    
class PSCONV_block(nn.Module):
    def __init__(self,inp,temp_ratio,expand_ratio,stride):
        super(PSCONV_block,self).__init__()
        self.inp=inp
        self.expand_ratio=expand_ratio
        self.stride=stride
        # self.use_connect=not (expand_ratio==1 and stride==1)
        self.use_connect=False
        self.conv=nn.Sequential(
            nn.Conv2d(inp,inp * temp_ratio,kernel_size=3,stride=stride,padding= 1,groups=inp,bias=False),
            nn.BatchNorm2d(inp * temp_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * temp_ratio,inp * temp_ratio,kernel_size=3,stride=1,padding= 1,groups=inp,bias=False),
            nn.BatchNorm2d(inp * temp_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * temp_ratio,inp * expand_ratio,kernel_size=1,stride=1,groups=inp,bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            )
        if self.use_connect:
            self.connect=nn.Sequential(
                nn.Conv2d(inp,inp * expand_ratio,kernel_size=3,stride=stride,padding= 1,groups=inp,bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                )

    def forward(self,x):
        if self.use_connect:
            return self.connect(x)+self.conv(x)
        else:
            return self.conv(x)

class SFC_NET_3_4_20(nn.Module):
    def __init__(self,num_classes=10):
        super(SFC_NET_3_4_20, self).__init__()
        self.num_classes=num_classes
        self.blocks=nn.ModuleList()
        C=20
        B=3
        E=2
        K=4
        inp=C
        self.stem=nn.Sequential(nn.Conv2d(3,inp,kernel_size=3, stride= 1,padding=1,bias=False),
            nn.BatchNorm2d(inp),)
        setting_list=[
        #temp_ratio  n stride
        # 
        [K,E,B,1],
        [K,E,B,2],
        [K,E,B,2],
        ]
        for temp_ratio,expand_ratio,n,stride in setting_list:
            stride_=[stride]+[1]*(n-1)
            expand_ratio_=[expand_ratio]+[1]*(n-1)
            for idx in range(n):
                self.blocks.append(PSCONV_block(inp,temp_ratio,1,stride_[idx]))
                oup=int(expand_ratio_[idx]*inp)
                self.blocks.append(conv_1x1_bn(inp,oup))
                inp=oup
        self.oup=inp
        self.global_pooling=nn.AdaptiveAvgPool2d(1)
        self.conv_classifier = nn.Sequential(
            nn.Linear(self.oup,num_classes,bias=False)
        )


    def forward(self,x):
        x=self.stem(x)
        for block in self.blocks:
            x=block(x)
        x=self.global_pooling(x)
        x=x.view(-1,self.oup)
        x=self.conv_classifier(x)

        return x
        




