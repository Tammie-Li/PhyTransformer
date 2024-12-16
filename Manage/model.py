'''
Author: Tammie li
Description: Define model
FilePath: \model.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pywt import wavedec
from sklearn import preprocessing

from layers.Embed import DataEmbedding

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if(self.max_norm != None):
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if(self.max_norm != None):
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

class MSOA(nn.Module):
    '''
    @description: multi-task self-supervised adaptation algorithm
    @inputparams: 
    n_class_main: Number of classes of main task
    n_class_aux_tem: Number of classes of auxiliary task VTO
    n_class_aux_spa: Number of classes of auxiliary task MSP
    channels:
    @Hyperparams: 
    n_kernel_t: Number of convolution kernels in temporal feature extraction block
    n_kernel_s: Number of convolution kernels in spatial feature extraction block
    dropout: default=0.5
    kernel_length: Length of  temporal convolution kernel, default=64
    '''
    def __init__(self, n_class_primary, T = 256, channels=64, n_kernel_t=8, n_kernel_s=16, dropout=0.5, kernel_length=32):
        super(MSOA, self).__init__()
        self.n_class_primary = n_class_primary
        self.channels = channels
        self.n_kernel_t = n_kernel_t
        self.n_kernel_s = n_kernel_s
        self.dropout = dropout
        self.kernel_length = kernel_length
   
        # Feature extractor (EEGNet Architecture)
        # self.temp_regular_conv = nn.Sequential(
        #     nn.ZeroPad2d((self.kernel_length//2-1, self.kernel_length//2, 0, 0)),
        #     nn.Conv2d(1, self.n_kernel_t, (1, self.kernel_length), bias=False),
        #     nn.BatchNorm2d(self.n_kernel_t)
        # )

        # self.spatial_depth_conv = nn.Sequential(
        #     Conv2dWithConstraint(self.n_kernel_t, self.n_kernel_s, (self.channels, 1), groups=self.n_kernel_t, bias=False),
        #     nn.BatchNorm2d(self.n_kernel_s),
        #     nn.ELU(), 
        #     nn.AvgPool2d((1, 4)),
        #     nn.Dropout(self.dropout)
        # )

        # self.fusion_point_conv = nn.Sequential(
        #     nn.ZeroPad2d((self.kernel_length//8-1, self.kernel_length//8, 0, 0)),
        #     nn.Conv2d(self.n_kernel_s, self.n_kernel_s, (1, self.kernel_length//4), groups=self.n_kernel_s, bias=False),
        #     nn.Conv2d(self.n_kernel_s, self.n_kernel_s, (1, 1), bias=False),
        #     nn.BatchNorm2d(self.n_kernel_s),
        #     nn.ELU(),
        #     nn.AvgPool2d((1, 8)),
        #     nn.Dropout(self.dropout)
        # )

        self.block0 = nn.Sequential()
        self.block0.add_module('conv1', nn.Conv2d(1, 25, (1, 5), bias=False))

        self.block1 = nn.Sequential()
        self.block1.add_module('conv2', nn.Conv2d(25, 25, (64, 1), bias=False))
        self.block1.add_module('norm1', nn.BatchNorm2d(25))
        self.block1.add_module('act1', nn.ELU())

        self.block1.add_module('pool1', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block1.add_module('drop1', nn.Dropout(p=0.5))

        self.block2 = nn.Sequential()

        self.block2.add_module('conv3', nn.Conv2d(25, 50, (1, 5), bias=False))
        self.block2.add_module('norm2', nn.BatchNorm2d(50))
        self.block2.add_module('act2', nn.ELU())

        self.block2.add_module('pool2', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block2.add_module('drop2', nn.Dropout(p=0.5))
        
        self.block3 = nn.Sequential()

        self.block3.add_module('conv4', nn.Conv2d(50, 100, (1, 5), bias=False))
        self.block3.add_module('norm3', nn.BatchNorm2d(100))
        self.block3.add_module('act3', nn.ELU())

        self.block3.add_module('pool3', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block3.add_module('drop3', nn.Dropout(p=0.5))

        self.block4 = nn.Sequential()

        self.block4.add_module('conv5', nn.Conv2d(100, 200, (1, 5), bias=False))
        self.block4.add_module('norm4', nn.BatchNorm2d(200))
        self.block4.add_module('act4', nn.ELU())

        self.block4.add_module('pool4', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block4.add_module('drop4', nn.Dropout(p=0.5))

        # Fully-connected layer
        # self.primary_task_classification = LinearWithConstraint(self.n_kernel_s*T//32, n_class_primary)
        # self.vto_task_projection_head = LinearWithConstraint(self.n_kernel_s*T//32, 16)
        # self.vto_task_classification = LinearWithConstraint(16, 9)

        # self.msp_task_projection_head = LinearWithConstraint(self.n_kernel_s*T//32, 16)
        # self.msp_task_classification = LinearWithConstraint(16, 8)

        self.primary_task_classification = nn.Linear(200*12, n_class_primary)
        self.vto_task_projection_head = nn.Linear(200*12, 16)
        self.vto_task_classification = nn.Linear(16, 9)

        self.msp_task_projection_head = nn.Linear(200*12, 16)
        self.msp_task_classification = nn.Linear(16, 8)


    def forward(self, x, stage_name, task_name):
        '''
        @description: Complete the corresponding task according to the task tag
        '''
        # extract features
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        
        # x = self.temp_regular_conv(x)
        # x = self.spatial_depth_conv(x)

        # x = self.fusion_point_conv(x)
        if stage_name == "testStageI":
            with torch.no_grad():
                x = self.block0(x)
        else:
            x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        fea = x.view(x.size(0), -1)

        if stage_name == "trainStage":
            if task_name == "main":
                logits = self.primary_task_classification(fea)
            elif task_name == "vto":
                logits = self.vto_task_projection_head(fea)
                logits = self.vto_task_classification(logits)
            elif task_name == "msp":
                logits = self.msp_task_projection_head(fea)
                logits = self.msp_task_classification(logits)
            else:
                assert("TaskName Error!")

            pred = F.softmax(logits, dim = 1)

            return pred

        elif stage_name == "testStageI":
            if task_name == "vto":
                with torch.no_grad():
                    logits = self.vto_task_projection_head(fea)
                    logits = self.vto_task_classification(logits)
            elif task_name == "msp":
                with torch.no_grad():
                    logits = self.msp_task_projection_head(fea)
                    logits = self.msp_task_classification(logits)
            else:
                assert("TaskName Error!")
            
            pred = F.softmax(logits, dim = 1)
            return pred

        elif stage_name == "testStageII":
            with torch.no_grad():
                primary = self.primary_task_classification(fea)
                pred = F.softmax(primary, dim = 1)
            return pred
        elif stage_name == "GetFeature":
            return fea        
        elif stage_name == "GetHeadVTO":
            head_vto = self.vto_task_projection_head(fea)
            return head_vto
        elif stage_name == "GetHeadMSP":
            head_msp = self.msp_task_projection_head(fea)
            return head_msp
        else:
            assert("Please enter the correct stage name!")


class DeepConvNet(nn.Module):
    def __init__(self, num_classes):
        super(DeepConvNet, self).__init__()
        dropout = 0.6
        self.block1 = nn.Sequential()
        self.block1.add_module('conv1', nn.Conv2d(1, 25, (1, 5), bias=False))

        self.block1.add_module('conv2', nn.Conv2d(25, 25, (64, 1), bias=False))
        self.block1.add_module('norm1', nn.BatchNorm2d(25))
        self.block1.add_module('act1', nn.ELU())

        self.block1.add_module('pool1', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block1.add_module('drop1', nn.Dropout(p=dropout))

        self.block2 = nn.Sequential()

        self.block2.add_module('conv3', nn.Conv2d(25, 50, (1, 5), bias=False))
        self.block2.add_module('norm2', nn.BatchNorm2d(50))
        self.block2.add_module('act2', nn.ELU())

        self.block2.add_module('pool2', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block2.add_module('drop2', nn.Dropout(p=dropout))
        
        self.block3 = nn.Sequential()

        self.block3.add_module('conv4', nn.Conv2d(50, 100, (1, 5), bias=False))
        self.block3.add_module('norm3', nn.BatchNorm2d(100))
        self.block3.add_module('act3', nn.ELU())

        self.block3.add_module('pool3', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block3.add_module('drop3', nn.Dropout(p=dropout))

        self.block4 = nn.Sequential()

        self.block4.add_module('conv5', nn.Conv2d(100, 200, (1, 5), bias=False))
        self.block4.add_module('norm4', nn.BatchNorm2d(200))
        self.block4.add_module('act4', nn.ELU())

        self.block4.add_module('pool4', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block4.add_module('drop4', nn.Dropout(p=dropout))

        self.classify = nn.Sequential(
            nn.Linear(200*12, num_classes)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        res = x.view(x.size(0), -1)

        out = self.classify(res)
        return out
    
class PPNN(nn.Module):
    def __init__(self, num_classes, F_t=8, F_s=8, T=256, C=64, drop_out=0.6):
        super(PPNN, self).__init__()
        self.F_t = F_t
        self.F_s = F_s
        self.T = T
        self.C = C
        self.drop_out = drop_out

        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((62, 62, 0, 0)),
            nn.Conv2d(1, self.F_t, (1, 3), dilation=(1, 2)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 4)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 8)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 16)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 32)),
        )
        self.conv = nn.Conv2d(1, self.F_t, (1, 1))

        self.block_2 = nn.Sequential(
            nn.BatchNorm2d(self.F_t),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.drop_out)
        )


        self.block_3 = nn.Sequential(
            nn.Conv2d(self.F_t, self.F_s, (self.C, 1)),
            nn.BatchNorm2d(self.F_s),
            nn.ELU(),
            nn.Dropout(self.drop_out)
        )

        self.fc = nn.Linear(self.F_s * self.T // 8, num_classes)

    def forward(self, x):
        # residual block
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        res = self.conv(x)
        x = self.block_1(x)
        x += res

        x = self.block_2(x)
        x = self.block_3(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

class EEGNet(nn.Module):
    def __init__(self, num_classes):
        super(EEGNet, self).__init__()
        self.drop_out = 0.4

        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8)
        )
        
        # block 2 and 3 are implements of Depthwise Conv and Separable Conv
        self.block_2 = nn.Sequential(
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.drop_out)
        )
        
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(16, 16, (1, 16), groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16), 
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.drop_out)
        )
        
        self.fc1 = nn.Linear((32 * (256 // 64)), num_classes)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.block_1(x)
        # fea = x

        x = self.block_2(x)
        x = self.block_3(x)

        x = x.view(x.size(0), -1)        

        logits = self.fc1(x)
        probas = F.softmax(logits, dim=1)
        return probas

class EEGInception(nn.Module):
    def __init__(self, num_classes, C=64, T=256, drop_out=0.5):
        super(EEGInception, self).__init__()
        self.T = T
        self.C = C
        self.drop_out = 0.5
        # input size: (N, 1, C, T)
        self.time_block_11 = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, 8, (1, 64)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out), 
            nn.Conv2d(8, 16, (self.C, 1), groups=8),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        self.time_block_12 = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out), 
            nn.Conv2d(8, 16, (self.C, 1), groups=8),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        self.time_block_13 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(1, 8, (1, 16)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out), 
            nn.Conv2d(8, 16, (self.C, 1), groups=8),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        
        self.time_block_21 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(48, 8, (1, 16)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        self.time_block_22 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(48, 8, (1, 8)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        self.time_block_23 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(48, 8, (1, 4)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        
        self.time_block_3 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(24, 12, (1, 8)),
            nn.BatchNorm2d(12),
            nn.Dropout(self.drop_out)
        )
        
        self.time_block_4 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(12, 6, (1, 4)),
            nn.BatchNorm2d(6),
            nn.Dropout(self.drop_out)
        )
        
        self.pool_1 = nn.AvgPool2d((1, 4))
        self.pool_2 = nn.AvgPool2d((1, 2))
        
        self.fc = nn.Linear(self.T // (4 * 2 * 2 * 2) * 6, num_classes)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x_11 = self.time_block_11(x)
        x_12 = self.time_block_12(x)
        x_13 = self.time_block_13(x)
        x = torch.cat((x_11, x_12, x_13), dim=1)
        x = self.pool_1(x)
        
        x_21 = self.time_block_21(x)
        x_22 = self.time_block_22(x)
        x_23 = self.time_block_23(x)
        x = torch.cat((x_21, x_22, x_23), dim=1)
        x = self.pool_2(x)
        
        x = self.time_block_3(x)
        x = self.pool_2(x)
        x = self.time_block_4(x)
        x = self.pool_2(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return probas

class DRL(nn.Module):
    def __init__(self, n_class=3, channels=64, f1=8, d=1, drop_out=0.5, kernel_length=3):
        super(DRL, self).__init__()
        self.F1 = f1
        self.drop_out = drop_out
        self.kernel_length = kernel_length
        self.D = d
        self.channel = channels

        self.temp_block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernel_length // 2,
                         self.kernel_length // 2, 0, 0)),
            nn.Conv2d(1, self.F1, (1, self.kernel_length), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        self.temp_block2 = nn.Sequential(
            nn.ZeroPad2d((self.kernel_length // 2,
                         self.kernel_length // 2, 0, 0)),
            nn.Conv2d(1, self.F1, (1, self.kernel_length), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        self.temp_block3 = nn.Sequential(
            nn.ZeroPad2d((self.kernel_length // 2,
                         self.kernel_length // 2, 0, 0)),
            nn.Conv2d(1, self.F1, (1, self.kernel_length), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        self.temp_block4 = nn.Sequential(
            nn.ZeroPad2d((self.kernel_length // 2,
                         self.kernel_length // 2, 0, 0)),
            nn.Conv2d(1, self.F1, (1, self.kernel_length), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        self.spatial_block1 = nn.Sequential(
            nn.Conv2d(self.F1, self.F1 * self.D, (self.channel, 1)),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.Dropout(self.drop_out)
        )

        self.spatial_block2 = nn.Sequential(
            nn.Conv2d(self.F1, self.F1 * self.D, (self.channel, 1)),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.Dropout(self.drop_out)
        )
        self.spatial_block3 = nn.Sequential(
            nn.Conv2d(self.F1, self.F1 * self.D, (self.channel, 1)),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.Dropout(self.drop_out)
        )
        self.spatial_block4 = nn.Sequential(
            nn.Conv2d(self.F1, self.F1 * self.D, (self.channel, 1)),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.Dropout(self.drop_out)
        )

        self.ts_conv1 = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 1))
        self.ts_conv2 = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 1))
        self.ts_conv3 = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 1))
        self.ts_conv4 = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 1))

        self.classifier = nn.Linear(32 * self.F1 * self.D, n_class)

    def forward(self, x):
        # divide into four levels of downsampling
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x1 = x
        x2 = x[:, :, :, range(0, x.shape[-1], 2)]
        x3 = x[:, :, :, range(0, x.shape[-1], 4)]
        x4 = x[:, :, :, range(0, x.shape[-1], 8)]

        x1 = self.temp_block1(x1)
        x2 = self.temp_block2(x2)
        x3 = self.temp_block3(x3)
        x4 = self.temp_block4(x4)

        x1 = self.spatial_block1(x1)
        x2 = self.spatial_block2(x2)
        x3 = self.spatial_block3(x3)
        x4 = self.spatial_block4(x4)

        x1 = self.ts_conv1(x1)
        x2 = self.ts_conv2(x2)
        x3 = self.ts_conv3(x3)
        x4 = self.ts_conv4(x4)

        x1 = x1[:, :, :, range(0, x.shape[-1], 2)]
        x = x1 + x2
        x = x[:, :, :, range(0, x.shape[-1], 2)]
        x = x + x3
        x = x[:, :, :, range(0, x.shape[-1], 2)]
        x = x + x4

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x

class PLNet(nn.Module):
    def __init__(self, num_classes):
        super(PLNet, self).__init__()
        self.block1 = torch.nn.Sequential(
            Conv2dWithConstraint(
                in_channels=1,
                out_channels=8,
                kernel_size=(1, 32),
                bias=False,
                stride=(1, 4),
                max_norm=0.5
            ),   # F1, C, T
            torch.nn.BatchNorm2d(
                num_features=8
            ),
            torch.nn.ELU()
        )

        tmp = torch.Tensor(np.ones((1, 1, 64, 256), dtype=float))
        tmp = self.block1(tmp)
        # Permute
        tmp = tmp.view(1, tmp.shape[3], tmp.shape[2], tmp.shape[1])

        self.block2 = torch.nn.Sequential(
            # DepthwiseConv2D
            Conv2dWithConstraint(
                in_channels=tmp.shape[1],
                out_channels=tmp.shape[1],
                kernel_size=(64, 1),
                max_norm=0.5,
                stride=1,
                groups=tmp.shape[1],
                bias=False
            )
        )
        tmp = self.block2(tmp)

        # Permute
        tmp = tmp.view(1, tmp.shape[3], tmp.shape[2], tmp.shape[1])
        self.block3 = nn.Sequential(
            nn.BatchNorm2d(
                num_features=8),
            nn.ELU(),
            nn.Dropout(0.5),
            # SeparableConv2D
            Conv2dWithConstraint(
                in_channels=tmp.shape[1],
                out_channels=tmp.shape[1] * 2,
                kernel_size=(1, 9),
                groups=tmp.shape[1],
                bias=False,
                stride=1,
                max_norm=0.5
            ),
            Conv2dWithConstraint(
                in_channels=tmp.shape[1] * 2,
                out_channels=tmp.shape[1] * 2,
                kernel_size=(1, 1),
                bias=False,
                stride=1,
                max_norm=0.5
            ),
            torch.nn.BatchNorm2d(
                num_features=tmp.shape[1] * 2),
            torch.nn.ELU(),
        )
        tmp = self.block3(tmp)
        self.pooling = torch.nn.AdaptiveAvgPool2d(1)
        tmp = self.pooling(tmp)
        self.drop_out = torch.nn.Dropout(0.5)

        self.classifier = torch.nn.Sequential(
            LinearWithConstraint(tmp.shape[1], num_classes, max_norm=0.1)
        )

    def forward(self, data):
        batch_size = len(data)
        data = data.reshape(data.shape[0], 1, data.shape[1], data.shape[2])
        data = self.block1(data)
        data = data.view(
            batch_size, data.shape[3], data.shape[2], data.shape[1])
        data = self.block2(data)
        data = data.view(
            batch_size, data.shape[3], data.shape[2], data.shape[1])
        data = self.block3(data)
        data = self.pooling(data)
        data = self.drop_out(data)
        data = data.view(batch_size, -1)
        return self.classifier(data)

class HDCA(nn.Module):
    def __init__(self, num_classes):
        super(HDCA, self).__init__()
        pass
    
    def forward(self, x):
        pass

class rLDA(nn.Module):
    def __init__(self, num_classes):
        super(rLDA, self).__init__()
        pass
    
    def forward(self, x):
        pass

class xDAWNRG(nn.Module):
    def __init__(self, num_classes):
        super(xDAWNRG, self).__init__()
        pass
    
    def forward(self, x):
        pass

class Transformer(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=1, num_heads=8, hidden_dim=256, dropout=0.5):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_dim * 8, num_classes)

        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8)
        )
        
    def forward(self, x, mask=None):
        # Input shape: (B, T, C)
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        x = self.block_1(x)

        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])
        
        x = x.permute(0, 2, 1)  # Shape: (B, C, T)

        x = self.embedding(x)   # Shape: (B, C, hidden_dim)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)  # Shape: (B, C, hidden_dim)

        x = x.reshape(x.shape[0] // 8, 8, x.shape[1], x.shape[2])

        x = x.mean(dim=3)  # Average over the channel dimension
        x = x.view(x.shape[0], -1)
        x = self.fc(x)     # Shape: (B, num_classes)
        return F.log_softmax(x, dim=-1)

class CP(nn.Module):
    def __init__(self):
        super(CP, self).__init__()
        self.drop_out = 0.5
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8)
        )
        
        # block 2 and 3 are implements of Depthwise Conv and Separable Conv
        self.block_2 = nn.Sequential(
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.drop_out)
        )
        
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(16, 16, (1, 16), groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16), 
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.drop_out)
        )
        self.classifier = nn.Linear((32 * (256 // 64)), 2)

    def forward(self, x, stage):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    
        if stage == "feature":
            x = self.block_1(x)
            x = self.block_2(x)
            x = self.block_3(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)      
            probas = F.softmax(x, dim=1)
        elif stage == "classifier":
            with torch.no_grad():
                x = self.block_1(x)
            x = self.block_2(x)
            x = self.block_3(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)      
            probas = F.softmax(x, dim=1)
        elif stage == "test":
            with torch.no_grad():
                x = self.block_1(x)
                x = self.block_2(x)
                x = self.block_3(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)      
                probas = F.softmax(x, dim=1)
        return probas    

class TC(nn.Module):
    def __init__(self):
        super(TC, self).__init__()
        self.drop_out = 0.5
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8)
        )
        
        # block 2 and 3 are implements of Depthwise Conv and Separable Conv
        self.block_2 = nn.Sequential(
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.drop_out)
        )
        
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(16, 16, (1, 16), groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16), 
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.drop_out)
        )
        self.classifier = nn.Linear((32 * (256 // 64)), 2)

    def forward(self, x, stage):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    
        if stage == "feature":
            x = self.block_1(x)
            x = self.block_2(x)
            x = self.block_3(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)      
            probas = F.softmax(x, dim=1)
        elif stage == "classifier":
            with torch.no_grad():
                x = self.block_1(x)
            x = self.block_2(x)
            x = self.block_3(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)      
            probas = F.softmax(x, dim=1)
        elif stage == "test":
            with torch.no_grad():
                x = self.block_1(x)
                x = self.block_2(x)
                x = self.block_3(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)      
                probas = F.softmax(x, dim=1)
        return probas    

class CPC(nn.Module):
    def __init__(self):
        super(CPC, self).__init__()
        self.drop_out = 0.5
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8)
        )
        
        # block 2 and 3 are implements of Depthwise Conv and Separable Conv
        self.block_2 = nn.Sequential(
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.drop_out)
        )
        
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(16, 16, (1, 16), groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16), 
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.drop_out)
        )
        self.proj_head_block = nn.Sequential()
        self.proj_head_block.add_module('fully connect layer 1', nn.Linear(128, 64))
        self.proj_head_block.add_module('activate function 1', nn.ELU())
        self.proj_head_block.add_module('fully connect layer 2', nn.Linear(64, 16))

        self.classifier = nn.Linear((32 * (256 // 64)), 2)

    def cal_loss(self, result, label):
        """
        Contrastive loss function 
        """
        sum_positive_pair = 0
        sum_negative_pair = 0
        for i in range(result.shape[0]//2):
            if label[i] == 0:
                sum_negative_pair += torch.exp(result[i])
            else:
                sum_positive_pair += torch.exp(result[i])
        sum_positive_pair += torch.exp(torch.tensor(-10, dtype=torch.float))
        sum_negative_pair += torch.exp(torch.tensor(-10, dtype=torch.float))
        loss = - torch.log(sum_negative_pair / sum_positive_pair)
        return loss

    def forward(self, x, stage):    

        if stage == "feature":
            result = torch.zeros((x.shape[0]))
            x1, x2 = x[:, 0, ...], x[:, 1, ...]

            x1 = x1.reshape(x1.shape[0], 1, x1.shape[1], x1.shape[2])
            x2 = x2.reshape(x2.shape[0], 1, x2.shape[1], x2.shape[2])
            x1, x2 = self.block_1(x1), self.block_1(x2)
            x1, x2 = self.block_2(x1), self.block_2(x2)
            x1, x2 = self.block_3(x1), self.block_3(x2)
            x1, x2 = x1.view(x1.size(0), -1), x2.view(x2.size(0), -1)
            x1, x2 = self.proj_head_block(x1), self.proj_head_block(x2)

            for i in range(result.shape[0]):
                similar = torch.cosine_similarity(x1[i, ...], x2[i, ...], dim=0)
                result[i] = similar
            return result
        
        elif stage == "classifier":
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
            with torch.no_grad():
                x = self.block_1(x)
            x = self.block_2(x)
            x = self.block_3(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)      
            probas = F.softmax(x, dim=1)
        elif stage == "test":
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
            with torch.no_grad():
                x = self.block_1(x)
                x = self.block_2(x)
                x = self.block_3(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)      
                probas = F.softmax(x, dim=1)
        return probas    

class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''
    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)          # B x 1 x E
        x = torch.cat([x, stats], dim=1) # B x 2 x E
        x = x.view(batch_size, -1) # B x 2E
        y = self.backbone(x)       # B x O

        return y

class MTCN(nn.Module):
    '''
    @description: Feature Decomposition for Reducing Negative Transfer: A Novel Multi-task Learning Method for Recommender System
    @inputparams: EEGNet

    @Hyperparams: 

    '''
    def __init__(self, n_class_primary, T = 256, channels=64, n_kernel_t=8, n_kernel_s=16, dropout=0.5, kernel_length=32):
        super(MTCN, self).__init__()

        self.n_class_primary = n_class_primary
        self.channels = channels
        self.n_kernel_t = n_kernel_t
        self.n_kernel_s = n_kernel_s
        self.dropout = dropout
        self.kernel_length = kernel_length

        
        self.block_shared_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )
        self.block_specific_main_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )
        self.block_specific_mtr_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )
        self.block_specific_msr_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )

        self.block_feature_fusion = nn.Sequential(
            nn.ZeroPad2d((self.kernel_length//8-1, self.kernel_length//8, 0, 0)),
            nn.Conv2d(self.n_kernel_s, self.n_kernel_s, (1, self.kernel_length//4), groups=self.n_kernel_s, bias=False),
            nn.Conv2d(self.n_kernel_s, self.n_kernel_s, (1, 1), bias=False),
            nn.BatchNorm2d(self.n_kernel_s),
            nn.ELU()
            # nn.AvgPool2d((1, 8)),
            # nn.Dropout(self.dropout)
        )
        self.main_task_projection_head =  nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropout)
        )
        self.vto_task_projection_head =  nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropout)
        )
        self.msp_task_projection_head =  nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropout)
        )

        # Fully-connected layer
        self.primary_task_classifier = nn.Sequential(
            nn.Linear(self.n_kernel_s*T//32, self.n_class_primary)
        )
        self.vto_task_classifier = nn.Sequential(
            nn.Linear(self.n_kernel_s*T//32, 9)
        )
        self.msp_task_classifier = nn.Sequential(
            nn.Linear(self.n_kernel_s*T//32, 8)
        )

    def calculate_orthogonal_constraint(self, feature_1, feature_2):
        assert feature_1.shape == feature_2.shape, "the dimension of two matrix is not equal"
        N, C, H, W = feature_1.shape
        feature_1, feature_2 = torch.reshape(feature_1, (N*C, H, W)), torch.reshape(feature_2, (N*C, H, W))
        weight_squared = torch.bmm(feature_1, feature_2.permute(0, 2, 1))
        # weight_squared = torch.norm(weight_squared, p=2)
        ones = torch.ones(N*C, H, H, dtype=torch.float32).to(torch.device('cuda:0'))
        diag = torch.eye(H, dtype=torch.float32).to(torch.device('cuda:0'))

        loss = ((weight_squared * (ones - diag)) ** 2).sum()
        return loss

    def forward(self, x, task_name):
        '''
        @description: Complete the corresponding task according to the task tag
        '''
        # extract features
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        
        fea_shared_extract = self.block_shared_feature_extractor(x)
        fea_after_fusion = self.block_feature_fusion(fea_shared_extract)

        if task_name == "main":
            # 推理过程
            fea_specific_main = self.block_specific_main_feature_extractor(x)
            fea_main = fea_specific_main + fea_after_fusion
            fea_main = self.main_task_projection_head(fea_main)
            fea_main = fea_main.view(fea_main.size(0), -1)
            logits_main = self.primary_task_classifier(fea_main)
            pred_main = F.softmax(logits_main, dim = 1)
            # 损失计算
            orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_main, fea_shared_extract)

            return pred_main, orthogonal_constraint

        elif task_name == "vto":
            # 推理过程
            fea_specific_vto = self.block_specific_mtr_feature_extractor(x)
            fea_vto = (fea_specific_vto + fea_after_fusion)
            fea_vto = self.vto_task_projection_head(fea_vto)
            fea_vto = fea_vto.view(fea_vto.size(0), -1)
            logits_vto = self.vto_task_classifier(fea_vto)
            pred_vto = F.softmax(logits_vto, dim = 1)
            # 损失计算
            orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_vto, fea_shared_extract)

            return pred_vto, orthogonal_constraint

        elif task_name == "msp":
            # 推理过程
            fea_specific_msp = self.block_specific_msr_feature_extractor(x)
            fea_msp = (fea_specific_msp + fea_after_fusion)
            fea_msp = self.msp_task_projection_head(fea_msp)
            fea_msp = fea_msp.view(fea_msp.size(0), -1)
            logits_msp = self.msp_task_classifier(fea_msp)
            pred_msp = F.softmax(logits_msp, dim = 1)
            # 损失计算
            orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_msp, fea_shared_extract)

            return pred_msp, orthogonal_constraint
        else:
            assert("TaskName Error!")

class Transform(nn.Module):
    """
    A Vertex Transformation module
    Permutation invariant transformation: (N, k, d) -> (N, k, d)
    """
    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super().__init__()

        self.convKK = nn.Conv1d(k, k * k, dim_in, groups=k)
        self.activation = nn.Softmax(dim=-1)
        self.dp = nn.Dropout()

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, k, d)
        """
        N, k, _ = region_feats.size()  # (N, k, d)
        conved = self.convKK(region_feats)  # (N, k*k, 1)
        multiplier = conved.view(N, k, k)  # (N, k, k)
        multiplier = self.activation(multiplier)  # softmax along last dimension
        transformed_feats = torch.matmul(multiplier, region_feats)  # (N, k, d)
        return transformed_feats

class VertexConv(nn.Module):
    """
    A Vertex Convolution layer
    Transform (N, k, d) feature to (N, d) feature by transform matrix and 1-D convolution
    """
    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super().__init__()

        self.trans = Transform(dim_in, k)                   # (N, k, d) -> (N, k, d)
        self.convK1 = nn.Conv1d(k, 1, 1)                    # (N, k, d) -> (N, 1, d)

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, d)
        """
        # transformed_feats = self.trans(region_feats)
        transformed_feats = region_feats
        pooled_feats = self.convK1(transformed_feats)             # (N, 1, d)
        pooled_feats = pooled_feats.squeeze(1)
        return pooled_feats

class EdgeConv(nn.Module):
    """
    A Hyperedge Convolution layer
    Using self-attention to aggregate hyperedges
    """
    def __init__(self, dim_ft, hidden):
        """
        :param dim_ft: feature dimension
        :param hidden: number of hidden layer neurons
        """
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim_ft, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, ft):
        """
        use self attention coefficient to compute weighted average on dim=-2
        :param ft (N, t, d)
        :return: y (N, d)
        """
        scores = []
        n_edges = ft.size(1)
        for i in range(n_edges):
            scores.append(self.fc(ft[:, i]))
        scores = torch.softmax(torch.stack(scores, 1), 1)
        
        return (scores * ft).sum(1)

class GCN_layer(nn.Module):

    def __init__(self, signal_shape, bias=False):
        super(GCN_layer, self).__init__()

        # input_shape=(node,timestep)
        self.W = nn.Parameter(torch.ones(signal_shape[0], signal_shape[0]), requires_grad=True)
        self.theta = nn.Parameter(torch.randn(signal_shape[1]), requires_grad=True)
        self.b = nn.Parameter(torch.zeros([1, 1, 1, signal_shape[1]]), requires_grad=True)
        self.bias = bias

    def forward(self, Adj_matrix, input_features):

        hadamard = Adj_matrix

        aggregate = torch.einsum("ce,abed->abcd", hadamard, input_features)
        output = torch.einsum("abcd,d->abcd", aggregate, self.theta)

        if self.bias == True:
            output = output + self.b

        return output

class EncoderLayerI(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(EncoderLayerI, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class EncoderI(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dropout=0.1):
        super(EncoderI, self).__init__()
        self.layers = nn.ModuleList([EncoderLayerI(d_model, n_heads, dropout) for _ in range(num_layers)])

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src
        return src

class Informer(nn.Module):
    def __init__(self, input_dim, num_classes, seq_len, d_model=256, n_heads=8, num_layers=1, dropout=0.3):
        super(Informer, self).__init__()
        self.seq_len = seq_len
        self.encoder = EncoderI(d_model, n_heads, num_layers, dropout)
        self.fc = nn.Linear(512, num_classes)

        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8)
        )

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        x = self.block_1(x)

        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])

        x = self.encoder(x)  # No need to transpose for compatibility with MultiheadAttention

        x = x.reshape(x.shape[0] // 8, 8, x.shape[1], x.shape[2])
        x = x.mean(dim=3)  # Average over the channel dimension
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(18920, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class ConvTransformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=1, n_classes=2, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(factor)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        
    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()

class PhyTransformer(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pad, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=(1, step), padding=(0, pad)),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.Dropout(0.4))
            # nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: 1 x EEG channel x datapoint
        super(PhyTransformer, self).__init__()
        self.stride = [1, 2, 2]
        self.kernel_size = 7
        self.pad = self.kernel_size // 2
        self.pool = 2
        self.F_t = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((62, 62, 0, 0)),
            nn.Conv2d(1, self.F_t, (1, 3), dilation=(1, 2)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 4)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 8)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 16)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 32)),
        )
        self.conv = nn.Conv2d(1, self.F_t, (1, 1))


        self.dropout = dropout_rate
        self.dmodel = 256
        self.n_heads = 8
        self.d_ff = 1024
        self.e_layers = 1
        self.use_norm = False
        # self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 5, attention_dropout=self.dropout,
                                      output_attention=False), self.dmodel, self.n_heads),
                    self.dmodel,
                    self.d_ff,
                    dropout=self.dropout,
                    activation="relu"
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.dmodel)
        )
        self.projector = nn.Linear(self.dmodel, 4, bias=True)

        # block 2 and 3 are implements of Depthwise Conv and Separable Conv
        self.block_2 = nn.Sequential(
            nn.Conv2d(8, 8, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.3)
        )
        
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(8, 8, (1, 8), groups=8, bias=False),
            nn.Conv2d(8, 8, (1, 1), bias=False),
            nn.BatchNorm2d(8), 
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.3)
        )
        
        self.fc1 = nn.Linear(64, num_classes)
    
    def _transform_EEG_channel(self, input):
        [N, C, H, W] = input.shape
        output = torch.zeros((N, C, 9, 9, W))

        for i in range(self._NeuralScan_Matric.shape[0]):
            for j in range(self._NeuralScan_Matric.shape[1]):
                if self._NeuralScan_Matric[i][j] == -1:
                    output[:, :, i, j, :] = 0
                else:
                    output[:, :, i, j, :] = input[:, :, self._NeuralScan_Matric[i][j], :]
        return output

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        # if self.use_norm:
        #     # Normalization from Non-stationary Transformer
        #     means = x.mean(1, keepdim=True).detach()
        #     x = x - means
        #     stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        #     x /= stdev
        # x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        # 时域信号提取
        # y1 = self.T_layer_1(x)
        # out = y1
        # y2 = self.T_layer_2(y1)
        # out = torch.cat((out, y2), dim=-1)

        # y3 = self.T_layer_3(y2)
        # out = torch.cat((out, y3), dim=-1)
        # out = out.permute(0, 1, 3, 2)
        res = self.conv(x)
        out = self.block_1(x)
        out += res


        out = out.reshape(out.shape[0]*out.shape[1], out.shape[2], out.shape[3])

        _, _, N = out.shape
        out, attns = self.encoder(out, attn_mask=None)

        out = out.reshape(out.shape[0] // 8, 8, out.shape[1], out.shape[2])


        out = self.block_2(out)
        out = self.block_3(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)    
        out = F.softmax(out, dim=1)

        return out


class PhyTransformer_T(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pad, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=(1, step), padding=(0, pad)),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.Dropout(0.4))
            # nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: 1 x EEG channel x datapoint
        super(PhyTransformer_T, self).__init__()
        self.stride = [1, 2, 2]
        self.kernel_size = 7
        self.pad = self.kernel_size // 2
        self.pool = 2
        self.F_t = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((62, 62, 0, 0)),
            nn.Conv2d(1, self.F_t, (1, 3), dilation=(1, 2)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 4)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 8)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 16)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 32)),
        )
        self.conv = nn.Conv2d(1, self.F_t, (1, 1))


        self.dropout = dropout_rate
        self.dmodel = 256
        self.n_heads = 8
        self.d_ff = 1024
        self.e_layers = 1
        self.use_norm = False
        # self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 5, attention_dropout=self.dropout,
                                      output_attention=False), self.dmodel, self.n_heads),
                    self.dmodel,
                    self.d_ff,
                    dropout=self.dropout,
                    activation="relu"
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.dmodel)
        )
        self.projector = nn.Linear(self.dmodel, 4, bias=True)

        # block 2 and 3 are implements of Depthwise Conv and Separable Conv
        self.block_2 = nn.Sequential(
            nn.Conv2d(8, 8, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.3)
        )
        
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(8, 8, (1, 8), groups=8, bias=False),
            nn.Conv2d(8, 8, (1, 1), bias=False),
            nn.BatchNorm2d(8), 
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.3)
        )
        
        self.fc1 = nn.Linear(64, num_classes)
    
    def _transform_EEG_channel(self, input):
        [N, C, H, W] = input.shape
        output = torch.zeros((N, C, 9, 9, W))

        for i in range(self._NeuralScan_Matric.shape[0]):
            for j in range(self._NeuralScan_Matric.shape[1]):
                if self._NeuralScan_Matric[i][j] == -1:
                    output[:, :, i, j, :] = 0
                else:
                    output[:, :, i, j, :] = input[:, :, self._NeuralScan_Matric[i][j], :]
        return output

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        # if self.use_norm:
        #     # Normalization from Non-stationary Transformer
        #     means = x.mean(1, keepdim=True).detach()
        #     x = x - means
        #     stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        #     x /= stdev
        # x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        # 时域信号提取
        # y1 = self.T_layer_1(x)
        # out = y1
        # y2 = self.T_layer_2(y1)
        # out = torch.cat((out, y2), dim=-1)

        # y3 = self.T_layer_3(y2)
        # out = torch.cat((out, y3), dim=-1)
        # out = out.permute(0, 1, 3, 2)

        # res = self.conv(x)
        # out = self.block_1(x)
        # out += res
        out = self.conv(x)


        out = out.reshape(out.shape[0]*out.shape[1], out.shape[2], out.shape[3])

        _, _, N = out.shape
        out, attns = self.encoder(out, attn_mask=None)

        out = out.reshape(out.shape[0] // 8, 8, out.shape[1], out.shape[2])


        out = self.block_2(out)
        out = self.block_3(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)    
        out = F.softmax(out, dim=1)

        return out

class PhyTransformer_S(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pad, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=(1, step), padding=(0, pad)),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.Dropout(0.4))
            # nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: 1 x EEG channel x datapoint
        super(PhyTransformer_S, self).__init__()

        self.stride = [1, 2, 2]
        self.kernel_size = 7
        self.pad = self.kernel_size // 2
        self.pool = 2
        self.F_t = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((62, 62, 0, 0)),
            nn.Conv2d(1, self.F_t, (1, 3), dilation=(1, 2)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 4)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 8)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 16)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 32)),
        )
        self.conv = nn.Conv2d(1, self.F_t, (1, 1))


        self.dropout = dropout_rate
        self.dmodel = 256
        self.n_heads = 8
        self.d_ff = 1024
        self.e_layers = 1
        self.use_norm = False
        # self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 5, attention_dropout=self.dropout,
                                      output_attention=False), self.dmodel, self.n_heads),
                    self.dmodel,
                    self.d_ff,
                    dropout=self.dropout,
                    activation="relu"
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.dmodel)
        )
        self.projector = nn.Linear(self.dmodel, 4, bias=True)

        # block 2 and 3 are implements of Depthwise Conv and Separable Conv
        self.block_2 = nn.Sequential(
            nn.Conv2d(8, 8, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.3)
        )
        
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(8, 8, (1, 8), groups=8, bias=False),
            nn.Conv2d(8, 8, (1, 1), bias=False),
            nn.BatchNorm2d(8), 
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.3)
        )
        
        self.fc1 = nn.Linear(64, num_classes)
    
    def _transform_EEG_channel(self, input):
        [N, C, H, W] = input.shape
        output = torch.zeros((N, C, 9, 9, W))

        for i in range(self._NeuralScan_Matric.shape[0]):
            for j in range(self._NeuralScan_Matric.shape[1]):
                if self._NeuralScan_Matric[i][j] == -1:
                    output[:, :, i, j, :] = 0
                else:
                    output[:, :, i, j, :] = input[:, :, self._NeuralScan_Matric[i][j], :]
        return output

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        # if self.use_norm:
        #     # Normalization from Non-stationary Transformer
        #     means = x.mean(1, keepdim=True).detach()
        #     x = x - means
        #     stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        #     x /= stdev
        # x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        # 时域信号提取
        # y1 = self.T_layer_1(x)
        # out = y1
        # y2 = self.T_layer_2(y1)
        # out = torch.cat((out, y2), dim=-1)

        # y3 = self.T_layer_3(y2)
        # out = torch.cat((out, y3), dim=-1)
        # out = out.permute(0, 1, 3, 2)

        res = self.conv(x)
        out = self.block_1(x)
        out += res


        # out = out.reshape(out.shape[0]*out.shape[1], out.shape[2], out.shape[3])

        # _, _, N = out.shape
        # out, attns = self.encoder(out, attn_mask=None)

        # out = out.reshape(out.shape[0] // 8, 8, out.shape[1], out.shape[2])


        out = self.block_2(out)
        out = self.block_3(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)    
        out = F.softmax(out, dim=1)

        return out

class PhyTransformer_C(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pad, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=(1, step), padding=(0, pad)),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.Dropout(0.4))
            # nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: 1 x EEG channel x datapoint
        super(PhyTransformer_C, self).__init__()
        self.stride = [1, 2, 2]
        self.kernel_size = 7
        self.pad = self.kernel_size // 2
        self.pool = 2
        self.F_t = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((62, 62, 0, 0)),
            nn.Conv2d(1, self.F_t, (1, 3), dilation=(1, 2)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 4)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 8)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 16)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 32)),
        )
        self.conv = nn.Conv2d(1, self.F_t, (1, 1))


        self.dropout = dropout_rate
        self.dmodel = 256
        self.n_heads = 8
        self.d_ff = 1024
        self.e_layers = 1
        self.use_norm = False
        # self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 5, attention_dropout=self.dropout,
                                      output_attention=False), self.dmodel, self.n_heads),
                    self.dmodel,
                    self.d_ff,
                    dropout=self.dropout,
                    activation="relu"
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.dmodel)
        )
        self.projector = nn.Linear(self.dmodel, 4, bias=True)

        # block 2 and 3 are implements of Depthwise Conv and Separable Conv
        self.block_2 = nn.Sequential(
            nn.Conv2d(8, 8, (64, 1), bias=False),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.3)
        )
        
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(8, 8, (1, 8), groups=8, bias=False),
            nn.Conv2d(8, 8, (1, 1), bias=False),
            nn.BatchNorm2d(8), 
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.3)
        )
        
        self.fc1 = nn.Linear(64, num_classes)
    
    def _transform_EEG_channel(self, input):
        [N, C, H, W] = input.shape
        output = torch.zeros((N, C, 9, 9, W))

        for i in range(self._NeuralScan_Matric.shape[0]):
            for j in range(self._NeuralScan_Matric.shape[1]):
                if self._NeuralScan_Matric[i][j] == -1:
                    output[:, :, i, j, :] = 0
                else:
                    output[:, :, i, j, :] = input[:, :, self._NeuralScan_Matric[i][j], :]
        return output

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        # if self.use_norm:
        #     # Normalization from Non-stationary Transformer
        #     means = x.mean(1, keepdim=True).detach()
        #     x = x - means
        #     stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        #     x /= stdev
        # x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        # 时域信号提取
        # y1 = self.T_layer_1(x)
        # out = y1
        # y2 = self.T_layer_2(y1)
        # out = torch.cat((out, y2), dim=-1)

        # y3 = self.T_layer_3(y2)
        # out = torch.cat((out, y3), dim=-1)
        # out = out.permute(0, 1, 3, 2)
        res = self.conv(x)
        out = self.block_1(x)
        out += res


        out = out.reshape(out.shape[0]*out.shape[1], out.shape[2], out.shape[3])

        _, _, N = out.shape
        out, attns = self.encoder(out, attn_mask=None)

        out = out.reshape(out.shape[0] // 8, 8, out.shape[1], out.shape[2])

        out = self.block_2(out)

        out = self.block_3(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)    

        out = F.softmax(out, dim=1)

        return out


        out = self.block_2(out)

class PhyTransformer_F(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pad, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=(1, step), padding=(0, pad)),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.Dropout(0.4))
            # nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: 1 x EEG channel x datapoint
        super(PhyTransformer_F, self).__init__()
        self.stride = [1, 2, 2]
        self.kernel_size = 7
        self.pad = self.kernel_size // 2
        self.pool = 2
        self.F_t = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((62, 62, 0, 0)),
            nn.Conv2d(1, self.F_t, (1, 3), dilation=(1, 2)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 4)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 8)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 16)),
            nn.Conv2d(self.F_t, self.F_t, (1, 3), dilation=(1, 32)),
        )
        self.conv = nn.Conv2d(1, self.F_t, (1, 1))


        self.dropout = dropout_rate
        self.dmodel = 256
        self.n_heads = 8
        self.d_ff = 1024
        self.e_layers = 1
        self.use_norm = False
        # self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 5, attention_dropout=self.dropout,
                                      output_attention=False), self.dmodel, self.n_heads),
                    self.dmodel,
                    self.d_ff,
                    dropout=self.dropout,
                    activation="relu"
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.dmodel)
        )
        self.projector = nn.Linear(self.dmodel, 4, bias=True)

        # block 2 and 3 are implements of Depthwise Conv and Separable Conv
        self.block_2 = nn.Sequential(
            nn.Conv2d(8, 8, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.3)
        )
        
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(8, 8, (1, 8), groups=8, bias=False),
            nn.Conv2d(8, 8, (1, 1), bias=False),
            nn.BatchNorm2d(8), 
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.3)
        )
        
        self.fc1 = nn.Linear(512, num_classes)
    
    def _transform_EEG_channel(self, input):
        [N, C, H, W] = input.shape
        output = torch.zeros((N, C, 9, 9, W))

        for i in range(self._NeuralScan_Matric.shape[0]):
            for j in range(self._NeuralScan_Matric.shape[1]):
                if self._NeuralScan_Matric[i][j] == -1:
                    output[:, :, i, j, :] = 0
                else:
                    output[:, :, i, j, :] = input[:, :, self._NeuralScan_Matric[i][j], :]
        return output

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        # if self.use_norm:
        #     # Normalization from Non-stationary Transformer
        #     means = x.mean(1, keepdim=True).detach()
        #     x = x - means
        #     stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        #     x /= stdev
        # x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        # 时域信号提取
        # y1 = self.T_layer_1(x)
        # out = y1
        # y2 = self.T_layer_2(y1)
        # out = torch.cat((out, y2), dim=-1)

        # y3 = self.T_layer_3(y2)
        # out = torch.cat((out, y3), dim=-1)
        # out = out.permute(0, 1, 3, 2)
        res = self.conv(x)
        out = self.block_1(x)
        out += res


        out = out.reshape(out.shape[0]*out.shape[1], out.shape[2], out.shape[3])

        _, _, N = out.shape
        out, attns = self.encoder(out, attn_mask=None)

        out = out.reshape(out.shape[0] // 8, 8, out.shape[1], out.shape[2])


        out = self.block_2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)    
        out = F.softmax(out, dim=1)

        return out

# from torch.nn.parameter import Parameter
# from torch.nn.modules.module import Module


# class GraphConvolution(Module):
#     """
#     simple GCN layer
#     """

#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
#         if bias:
#             self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
#         else:
#             self.register_parameter('bias', None)

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, x, adj):
#         output = torch.matmul(x, self.weight)-self.bias
#         output = F.relu(torch.matmul(adj, output))
#         return output

# class PowerLayer(nn.Module):
#     '''
#     The power layer: calculates the log-transformed power of the data
#     '''
#     def __init__(self, dim, length, step):
#         super(PowerLayer, self).__init__()
#         self.dim = dim
#         self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

#     def forward(self, x):
#         return torch.log(self.pooling(x.pow(2)))


# class DH2GNN_2(nn.Module):
#     def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
#         return nn.Sequential(
#             nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
#             PowerLayer(dim=-1, length=pool, step=int(pool_step_rate*pool))
#         )

#     def __init__(self, num_classes, input_size, sampling_rate, num_T,
#                  out_graph, dropout_rate, pool, pool_step_rate, idx_graph):
#         # input_size: EEG frequency x channel x datapoint
#         super(DH2GNN_2, self).__init__()
#         self.idx = idx_graph
#         self.window = [0.5, 0.25, 0.125]
#         self.pool = pool
#         self.channel = input_size[1]
#         self.brain_area = len(self.idx)

#         # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
#         # achieve the 1d convolution operation
#         self.Tception1 = self.temporal_learner(input_size[0], num_T,
#                                                (1, int(self.window[0] * sampling_rate)),
#                                                self.pool, pool_step_rate)
#         self.Tception2 = self.temporal_learner(input_size[0], num_T,
#                                                (1, int(self.window[1] * sampling_rate)),
#                                                self.pool, pool_step_rate)
#         self.Tception3 = self.temporal_learner(input_size[0], num_T,
#                                                (1, int(self.window[2] * sampling_rate)),
#                                                self.pool, pool_step_rate)
#         self.BN_t = nn.BatchNorm2d(num_T)
#         self.BN_t_ = nn.BatchNorm2d(num_T)
#         self.OneXOneConv = nn.Sequential(
#             nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
#             nn.LeakyReLU(),
#             nn.AvgPool2d((1, 2)))
#         # diag(W) to assign a weight to each local areas
#         size = self.get_size_temporal(input_size)
#         self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, size[-1]),
#                                                 requires_grad=True)
#         nn.init.xavier_uniform_(self.local_filter_weight)
#         self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
#                                               requires_grad=True)

#         # aggregate function
#         self.aggregate = Aggregator(self.idx)

#         # trainable adj weight for global network
#         self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
#         nn.init.xavier_uniform_(self.global_adj)
#         # to be used after local graph embedding
#         self.bn = nn.BatchNorm1d(self.brain_area)
#         self.bn_ = nn.BatchNorm1d(self.brain_area)
#         # learn the global network of networks
#         self.GCN = GraphConvolution(size[-1], out_graph)

#         self.fc = nn.Sequential(
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(int(self.brain_area * out_graph), num_classes))

#     def forward(self, x):
#         y = self.Tception1(x)
#         out = y
#         y = self.Tception2(x)
#         out = torch.cat((out, y), dim=-1)
#         y = self.Tception3(x)
#         out = torch.cat((out, y), dim=-1)
#         out = self.BN_t(out)
#         out = self.OneXOneConv(out)
#         out = self.BN_t_(out)
#         out = out.permute(0, 2, 1, 3)
#         out = torch.reshape(out, (out.size(0), out.size(1), -1))
#         out = self.local_filter_fun(out, self.local_filter_weight)
#         out = self.aggregate.forward(out)
#         adj = self.get_adj(out)
#         out = self.bn(out)
#         out = self.GCN(out, adj)
#         out = self.bn_(out)
#         out = out.view(out.size()[0], -1)
#         out = self.fc(out)
#         return out

#     def get_size_temporal(self, input_size):
#         # input_size: frequency x channel x data point
#         data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
#         z = self.Tception1(data)
#         out = z
#         z = self.Tception2(data)
#         out = torch.cat((out, z), dim=-1)
#         z = self.Tception3(data)
#         out = torch.cat((out, z), dim=-1)
#         out = self.BN_t(out)
#         out = self.OneXOneConv(out)
#         out = self.BN_t_(out)
#         out = out.permute(0, 2, 1, 3)
#         out = torch.reshape(out, (out.size(0), out.size(1), -1))
#         size = out.size()
#         return size

#     def local_filter_fun(self, x, w):
#         w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
#         x = F.relu(torch.mul(x, w) - self.local_filter_bias)
#         return x

#     def get_adj(self, x, self_loop=True):
#         # x: b, node, feature
#         adj = self.self_similarity(x)   # b, n, n
#         num_nodes = adj.shape[-1]
#         adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
#         if self_loop:
#             adj = adj + torch.eye(num_nodes).to(DEVICE)
#         rowsum = torch.sum(adj, dim=-1)
#         mask = torch.zeros_like(rowsum)
#         mask[rowsum == 0] = 1
#         rowsum += mask
#         d_inv_sqrt = torch.pow(rowsum, -0.5)
#         d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
#         adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
#         return adj

#     def self_similarity(self, x):
#         # x: b, node, feature
#         x_ = x.permute(0, 2, 1)
#         s = torch.bmm(x, x_)
#         return s


# class Aggregator():

#     def __init__(self, idx_area):
#         # chan_in_area: a list of the number of channels within each area
#         self.chan_in_area = idx_area
#         self.idx = self.get_idx(idx_area)
#         self.area = len(idx_area)

#     def forward(self, x):
#         # x: batch x channel x data
#         data = []
#         for i, area in enumerate(range(self.area)):
#             if i < self.area - 1:
#                 data.append(self.aggr_fun(x[:, self.idx[i]:self.idx[i + 1], :], dim=1))
#             else:
#                 data.append(self.aggr_fun(x[:, self.idx[i]:, :], dim=1))
#         return torch.stack(data, dim=1)

#     def get_idx(self, chan_in_area):
#         idx = [0] + chan_in_area
#         idx_ = [0]
#         for i in idx:
#             idx_.append(idx_[-1] + i)
#         return idx_[1:]

#     def aggr_fun(self, x, dim):
#         # return torch.max(x, dim=dim).values
#         return torch.mean(x, dim=dim)

# class FDN(nn.Module):
#     '''
#     @description: Feature Decomposition for Reducing Negative Transfer: A Novel Multi-task Learning Method for Recommender System
#     @inputparams:PLNet

#     @Hyperparams: 

#     '''
#     def __init__(self, n_class_primary, T = 256, channels=64, n_kernel_t=8, n_kernel_s=16, dropout=0.33, kernel_length=32):
#         super(FDN, self).__init__()

#         self.n_class_primary = n_class_primary
#         self.channels = channels
#         self.n_kernel_t = n_kernel_t
#         self.n_kernel_s = n_kernel_s
#         self.dropout = dropout
#         self.kernel_length = kernel_length

        
#         self.block_shared_feature_extractor_1 = nn.Sequential(
#             Conv2dWithConstraint(in_channels=1, out_channels=8, kernel_size=(1, 32), bias=False, stride=(1, 4), max_norm=0.5), 
#             torch.nn.BatchNorm2d(num_features=8),
#             torch.nn.ELU(),
#         )
              
#         tmp_shared_1 = torch.Tensor(np.ones((1, 1, 64, 256), dtype=float))
#         tmp_shared_1 = self.block_shared_feature_extractor_1(tmp_shared_1)
#         # Permute
#         tmp_shared_1 = tmp_shared_1.view(1, tmp_shared_1.shape[3], tmp_shared_1.shape[2], tmp_shared_1.shape[1])
        
#         self.block_shared_feature_extractor_2 = torch.nn.Sequential(
#             Conv2dWithConstraint(in_channels=tmp_shared_1.shape[1], out_channels=tmp_shared_1.shape[1], 
#                                  kernel_size=(64, 1), max_norm=0.5, stride=1, groups=tmp_shared_1.shape[1], bias=False)
#         )
#         tmp_shared_2 = self.block_shared_feature_extractor_2(tmp_shared_1)
#         tmp_shared_2 = tmp_shared_2.view(1, tmp_shared_2.shape[3], tmp_shared_2.shape[2], tmp_shared_2.shape[1])

#         # main
#         self.block_specific_main_feature_extractor_1 = nn.Sequential(
#             Conv2dWithConstraint(in_channels=1, out_channels=8, kernel_size=(1, 32), bias=False, stride=(1, 4), max_norm=0.5), 
#             torch.nn.BatchNorm2d(num_features=8),
#             torch.nn.ELU(),
#         )
              
#         tmp_specific_main_1 = torch.Tensor(np.ones((1, 1, 64, 256), dtype=float))
#         tmp_specific_main_1 = self.block_specific_main_feature_extractor_1(tmp_specific_main_1)
#         # Permute
#         tmp_specific_main_1 = tmp_specific_main_1.view(1, tmp_specific_main_1.shape[3], tmp_specific_main_1.shape[2], tmp_specific_main_1.shape[1])
        
#         self.block_specific_main_feature_extractor_2 = torch.nn.Sequential(
#             Conv2dWithConstraint(in_channels=tmp_specific_main_1.shape[1], out_channels=tmp_specific_main_1.shape[1], 
#                                  kernel_size=(64, 1), max_norm=0.5, stride=1, groups=tmp_specific_main_1.shape[1], bias=False)
#         )
#         tmp_specific_main_2 = self.block_specific_main_feature_extractor_2(tmp_specific_main_1)
#         tmp_specific_main_2 = tmp_specific_main_2.view(1, tmp_specific_main_2.shape[3], tmp_specific_main_2.shape[2], tmp_specific_main_2.shape[1])

#         # vto
#         self.block_specific_vto_feature_extractor_1 = nn.Sequential(
#             Conv2dWithConstraint(in_channels=1, out_channels=8, kernel_size=(1, 32), bias=False, stride=(1, 4), max_norm=0.5), 
#             torch.nn.BatchNorm2d(num_features=8),
#             torch.nn.ELU(),
#         )
              
#         tmp_specific_vto_1 = torch.Tensor(np.ones((1, 1, 64, 256), dtype=float))
#         tmp_specific_vto_1 = self.block_specific_vto_feature_extractor_1(tmp_specific_vto_1)
#         # Permute
#         tmp_specific_vto_1 = tmp_specific_vto_1.view(1, tmp_specific_vto_1.shape[3], tmp_specific_vto_1.shape[2], tmp_specific_vto_1.shape[1])
        
#         self.block_specific_vto_feature_extractor_2 = torch.nn.Sequential(
#             Conv2dWithConstraint(in_channels=tmp_specific_vto_1.shape[1], out_channels=tmp_specific_vto_1.shape[1], 
#                                  kernel_size=(64, 1), max_norm=0.5, stride=1, groups=tmp_specific_vto_1.shape[1], bias=False)
#         )
#         tmp_specific_vto_2 = self.block_specific_vto_feature_extractor_2(tmp_specific_vto_1)
#         tmp_specific_vto_2 = tmp_specific_vto_2.view(1, tmp_specific_vto_2.shape[3], tmp_specific_vto_2.shape[2], tmp_specific_vto_2.shape[1])

#         # msp
#         self.block_specific_msp_feature_extractor_1 = nn.Sequential(
#             Conv2dWithConstraint(in_channels=1, out_channels=8, kernel_size=(1, 32), bias=False, stride=(1, 4), max_norm=0.5), 
#             torch.nn.BatchNorm2d(num_features=8),
#             torch.nn.ELU(),
#         )
              
#         tmp_specific_msp_1 = torch.Tensor(np.ones((1, 1, 64, 256), dtype=float))
#         tmp_specific_msp_1 = self.block_specific_msp_feature_extractor_1(tmp_specific_msp_1)
#         # Permute
#         tmp_specific_msp_1 = tmp_specific_msp_1.view(1, tmp_specific_msp_1.shape[3], tmp_specific_msp_1.shape[2], tmp_specific_msp_1.shape[1])
        
#         self.block_specific_msp_feature_extractor_2 = torch.nn.Sequential(
#             Conv2dWithConstraint(in_channels=tmp_specific_msp_1.shape[1], out_channels=tmp_specific_msp_1.shape[1], 
#                                  kernel_size=(64, 1), max_norm=0.5, stride=1, groups=tmp_specific_msp_1.shape[1], bias=False)
#         )
#         tmp_specific_msp_2 = self.block_specific_msp_feature_extractor_2(tmp_specific_msp_1)
#         tmp_specific_msp_2 = tmp_specific_msp_2.view(1, tmp_specific_msp_2.shape[3], tmp_specific_msp_2.shape[2], tmp_specific_msp_2.shape[1])
        
#         tmp_specific_msp_2 = torch.cat([tmp_specific_msp_2, tmp_specific_msp_2], dim=1)
#         self.block_feature_fusion = nn.Sequential(
#             nn.BatchNorm2d(num_features=16),
#             nn.ELU(),
#             nn.Dropout(0.5),
#             # SeparableConv2D
#             Conv2dWithConstraint(in_channels=tmp_specific_msp_2.shape[1], out_channels=tmp_specific_msp_2.shape[1] * 2, 
#                                  kernel_size=(1, 9), groups=tmp_specific_msp_2.shape[1], bias=False, stride=1, max_norm=0.5),
#             Conv2dWithConstraint(in_channels=tmp_specific_msp_2.shape[1] * 2, out_channels=tmp_specific_msp_2.shape[1] * 2, 
#                                  kernel_size=(1, 1), bias=False, stride=1, max_norm=0.),
#             torch.nn.BatchNorm2d(num_features=tmp_specific_msp_2.shape[1] * 2),
#             torch.nn.ELU(),
#         )
#         fusion_feature = self.block_feature_fusion(tmp_specific_msp_2)

#         # fea_after_fusion = self.pooling(fea_after_fusion)
#         # fea_after_fusion = self.drop_out(fea_after_fusion)

#         self.main_task_projection_head =  nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Dropout(0.5)
#         )
#         self.vto_task_projection_head =  nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Dropout(0.5)
#         )
#         self.msp_task_projection_head =  nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Dropout(0.5)
#         )

#         # Fully-connected layer
#         self.primary_task_classifier = nn.Sequential(
#             nn.Linear(fusion_feature.shape[1], n_class_primary)
#         )
#         self.vto_task_classifier = nn.Sequential(
#             nn.Linear(fusion_feature.shape[1], 9)
#         )
#         self.msp_task_classifier = nn.Sequential(
#             nn.Linear(fusion_feature.shape[1], 8)
#         )

#     def calculate_orthogonal_constraint(self, feature_1, feature_2):
#         assert feature_1.shape == feature_2.shape, "the dimension of two matrix is not equal"
#         N, C, H, W = feature_1.shape
#         feature_1, feature_2 = torch.reshape(feature_1, (N*C, H, W)), torch.reshape(feature_2, (N*C, H, W))
#         weight_squared = torch.bmm(feature_1, feature_2.permute(0, 2, 1))
#         # weight_squared = torch.norm(weight_squared, p=2)
#         ones = torch.ones(N*C, H, H, dtype=torch.float32).to(torch.device('cuda:0'))
#         diag = torch.eye(H, dtype=torch.float32).to(torch.device('cuda:0'))

#         loss = ((weight_squared * (ones - diag)) ** 2).sum()
#         return loss

#     def forward(self, x, task_name):
#         '''
#         @description: Complete the corresponding task according to the task tag
#         '''
#         # extract features

#         batch_size = len(x)
#         fea_shared_extract = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
#         fea_shared_extract = self.block_shared_feature_extractor_1(fea_shared_extract)
#         fea_shared_extract = fea_shared_extract.view(batch_size, fea_shared_extract.shape[3], fea_shared_extract.shape[2], fea_shared_extract.shape[1])
#         fea_shared_extract = self.block_shared_feature_extractor_2(fea_shared_extract)
#         fea_shared_extract = fea_shared_extract.view(batch_size, fea_shared_extract.shape[3], fea_shared_extract.shape[2], fea_shared_extract.shape[1])

#         # fea_after_fusion = self.block_feature_fusion(fea_shared_extract)

        
#         if task_name == "main":
#             # 推理过程
#             x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
#             x = self.block_specific_main_feature_extractor_1(x)
#             x = x.view(batch_size, x.shape[3], x.shape[2], x.shape[1])
#             x = self.block_specific_main_feature_extractor_2(x)
#             fea_specific_main = x.view(batch_size, x.shape[3], x.shape[2], x.shape[1])

#             # print(fea_specific_main.shape, fea_shared_extract.shape)
#             # fea_main = fea_specific_main + fea_shared_extract
#             fea_main = torch.cat([fea_specific_main, fea_shared_extract], dim=1)

#             fea_main = self.block_feature_fusion(fea_main)
#             fea_main = self.main_task_projection_head(fea_main)
#             fea_main = fea_main.view(fea_main.size(0), -1)
#             logits_main = self.primary_task_classifier(fea_main)
#             pred_main = F.softmax(logits_main, dim = 1)
#             # 损失计算
#             orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_main, fea_shared_extract)

#             return pred_main, orthogonal_constraint

#         elif task_name == "vto":
#             # 推理过程
#             x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
#             x = self.block_specific_vto_feature_extractor_1(x)
#             x = x.view(batch_size, x.shape[3], x.shape[2], x.shape[1])
#             x = self.block_specific_vto_feature_extractor_2(x)
#             fea_specific_vto = x.view(batch_size, x.shape[3], x.shape[2], x.shape[1])
#             # fea_vto = fea_specific_vto + fea_shared_extract
#             fea_vto = torch.cat([fea_specific_vto, fea_shared_extract], dim=1)
#             fea_vto = self.block_feature_fusion(fea_vto)
#             fea_vto = self.vto_task_projection_head(fea_vto)

#             fea_vto = fea_vto.view(fea_vto.size(0), -1)
#             logits_vto = self.vto_task_classifier(fea_vto)
#             pred_vto = F.softmax(logits_vto, dim = 1)
#             # 损失计算
#             orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_vto, fea_shared_extract)

#             return pred_vto, orthogonal_constraint

#         elif task_name == "msp":
#             # 推理过程
#             x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
#             x = self.block_specific_msp_feature_extractor_1(x)
#             x = x.view(batch_size, x.shape[3], x.shape[2], x.shape[1])
#             x = self.block_specific_msp_feature_extractor_2(x)
#             fea_specific_msp = x.view(batch_size, x.shape[3], x.shape[2], x.shape[1])
#             # fea_msp = fea_specific_msp + fea_shared_extract
#             fea_msp = torch.cat([fea_specific_msp, fea_shared_extract], dim=1)
#             fea_msp = self.block_feature_fusion(fea_msp)
#             fea_msp = self.msp_task_projection_head(fea_msp)

#             fea_msp = fea_msp.view(fea_msp.size(0), -1)
#             logits_msp = self.msp_task_classifier(fea_msp)
#             pred_msp = F.softmax(logits_msp, dim = 1)
#             # 损失计算
#             orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_msp, fea_shared_extract)

#             return pred_msp, orthogonal_constraint
#         else:
#             assert("TaskName Error!")

class RSVPTransform(nn.Module):
    """
    Non-stationary Transformer
    """
    def __init__(self, configs):
        super(RSVPTransform, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.tau_learner   = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=configs.seq_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach() # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x E
        x_enc = x_enc / std_enc
        x_dec_new = torch.cat([x_enc[:, -self.label_len: , :], torch.zeros_like(x_dec[:, -self.pred_len:, :])], dim=1).to(x_enc.device).clone()

        tau = self.tau_learner(x_raw, std_enc).exp()     # B x S x E, B x 1 x E -> B x 1, positive scalar    
        delta = self.delta_learner(x_raw, mean_enc)      # B x S x E, B x 1 x E -> B x S

        # Model Inference
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)

        # De-normalization
        dec_out = dec_out * std_enc + mean_enc

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

if __name__ == "__main__":
    data = torch.tensor(np.random.rand(64, 32, 256)).to(torch.float32)
    model = ConvTransformer()

    a = model(data, "main")

