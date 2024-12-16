from re import S
from scipy.sparse.lil import _prepare_index_for_memoryview
from scipy.sparse.linalg.interface import _PowerLinearOperator
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.manifold import TSNE

import torch.nn.functional as F
import torch
import torch.nn as nn
import gc

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    def forward(self, x, task):
        '''
        @description: Complete the corresponding task according to the task tag
        '''
        # extract features

        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        
        fea_shared_extract = self.block_shared_feature_extractor(x)
        fea_after_fusion = self.block_feature_fusion(fea_shared_extract)

        if task == "MTR":
            # 推理过程
            fea_specific_vto = self.block_specific_mtr_feature_extractor(x)
            fea_vto = (fea_specific_vto + fea_after_fusion)
            fea_vto = self.vto_task_projection_head(fea_vto)
            fea_vto = fea_vto.view(fea_vto.size(0), -1)
            # logits_vto = self.vto_task_classifier(fea_vto)
            # pred_vto = F.softmax(logits_vto, dim = 1)
            del fea_shared_extract, fea_after_fusion, x, fea_specific_vto
            gc.collect()
            return fea_vto
        
        elif task == "MSR":
            fea_specific_msp = self.block_specific_msr_feature_extractor(x)
            fea_msp = (fea_specific_msp + fea_after_fusion)
            fea_msp = self.msp_task_projection_head(fea_msp)
            fea_msp = fea_msp.view(fea_msp.size(0), -1)
            # logits_msp = self.msp_task_classifier(fea_msp)
            # pred_msp = F.softmax(logits_msp, dim = 1)
            del fea_shared_extract, fea_after_fusion, x, fea_specific_msp
            gc.collect()
            return fea_msp
        
        elif task == "main":
            fea_specific_main = self.block_specific_main_feature_extractor(x)
            fea_main = fea_specific_main + fea_after_fusion
            fea_main = self.main_task_projection_head(fea_main)
            fea_main = fea_main.view(fea_main.size(0), -1)
            # logits_main = self.primary_task_classifier(fea_main)
            # pred_main = F.softmax(logits_main, dim = 1)
            del fea_shared_extract, fea_after_fusion, x, fea_specific_main
            gc.collect()
            return fea_main

def calculate_feature(param, data, task):

    model = MTCN(n_class_primary=2).to(DEVICE)
    model.load_state_dict(param)
    model.eval()
    x = torch.tensor(data).to(torch.float32)
    x = x.to(DEVICE)
    feature = model(x, task)
    del x, param, data
    gc.collect()
    return feature

def load_feature(dataset_name, sub_id, task_name):
    params_dir = os.path.join(os.getcwd(), "ExperimentForMTCN", f"{dataset_name}", "Params")
    data_dir = os.path.join(os.getcwd(), "Dataset", f"{dataset_name}")
    y_pred, y = [], []

    param = torch.load(os.path.join(params_dir, f"{dataset_name}_MTCN_{sub_id:>02d}.pth"))
    if task_name == "MTR":
        data = np.load(os.path.join(data_dir, f"S{sub_id:>02d}", "x_mtr_train.npy"))
        label = np.load(os.path.join(data_dir, f"S{sub_id:>02d}", "y_mtr_train.npy"))
    elif task_name == "MSR":
        data = np.load(os.path.join(data_dir, f"S{sub_id:>02d}", "x_msr_train.npy"))
        label = np.load(os.path.join(data_dir, f"S{sub_id:>02d}", "y_msr_train.npy"))  
    elif task_name == "main":
        data = np.load(os.path.join(data_dir, f"S{sub_id:>02d}", "x_train.npy"))
        label = np.load(os.path.join(data_dir, f"S{sub_id:>02d}", "y_train.npy"))  

    value = data.shape[0] // 2
    data1 = data[:value, ...]
    data2 = data[value:, ...]

    feature1 = calculate_feature(param, data1, task_name).cpu().detach()
    feature2 = calculate_feature(param, data2, task_name).cpu().detach()

    feature = np.concatenate((feature1, feature2))

    return feature, label


def t_SNE_transfer(data):
    # data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    tsne = TSNE(n_components=2)
    tsne.fit_transform(data)

    print("stage 2: data_transfer_success!")
    return tsne.embedding_

def construct_data_frame(data, label, task):
    tb_feature = pd.DataFrame(data, columns=['x1','x2'])
    label[0] = 0 
    label_t = []
    print(label)
    size = []
    if task == "main":
        for i in range(label.shape[0]):
            # size.append(50)
            if label[i] == 1:
                label_t.append('non-target')
            else:
                label_t.append('target')
    elif task == "MTR":
        for i in range(label.shape[0]):
            # size.append(50)
            if label[i] == 0: label_t.append('C1')
            if label[i] == 1: label_t.append('vP1')
            if label[i] == 2: label_t.append('N1')
            if label[i] == 3: label_t.append('vN1')
            if label[i] == 4: label_t.append('P2')
            if label[i] == 5: label_t.append('vN2')
            if label[i] == 6: label_t.append('P3')
            if label[i] == 7: label_t.append('N3')
            if label[i] == 8: label_t.append('SW')
    elif task == "MSR":
        for i in range(label.shape[0]):
            # size.append(50)
            if label[i] == 0: label_t.append('PF')
            if label[i] == 1: label_t.append('LT')
            if label[i] == 2: label_t.append('F')
            if label[i] == 3: label_t.append('RT')
            if label[i] == 4: label_t.append('LP')
            if label[i] == 5: label_t.append('C')
            if label[i] == 6: label_t.append('RP')
            if label[i] == 7: label_t.append('O')

    tb_feature = pd.DataFrame(data, columns=['x1','x2'])
    tb_feature['class'] = np.array(label_t)

    print("stage 3: construct_data_frame_success!")
    return tb_feature


def plot_t_SNE(data_frame, subject_id, task):
    sns.set_theme(style='white', font='Times New Roman', font_scale=1.3)
    color=['red','blue','pink','green','black']
    current_palette = sns.color_palette(color)
    # plt.figure(figsize=(10, 6.18))
    sns.set_palette(current_palette)
    g = sns.scatterplot(data=data_frame, x="x1", y="x2", hue="class", style='class', size='class', legend=False, edgecolors=None)
    g.xaxis.label.set_visible(False)
    g.yaxis.label.set_visible(False)
    print("stage 4: plot_t_SNE_success!")
    # plt.title(f'S{subject_id:>02d} raw data')
    # plt.xlim(-80, 80)
    # plt.ylim(-80, 80)
    # plt.show()
    scatter_fig = g.get_figure()
    plt.show()
    scatter_fig.savefig(f'{task}_S{subject_id:>02d}_feature.png', dpi = 600, bbox_inches='tight')


if __name__ == '__main__':
    _subject_id = [1, 2, 3]
    DATASET = "THU"
    task = ["main", "MTR", "MSR"]

    for subject_id in _subject_id:
        print(f"=================S{subject_id:>02d}=================")
        for j in task:
            data, label = load_feature(DATASET, subject_id, j)
            data = t_SNE_transfer(data)
            dataFrame = construct_data_frame(data, label, j)
            plot_t_SNE(dataFrame, subject_id, j)
