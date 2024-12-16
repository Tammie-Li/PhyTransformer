# confusion_matrix
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix
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
            logits_vto = self.vto_task_classifier(fea_vto)
            pred_vto = F.softmax(logits_vto, dim = 1)

            del fea_shared_extract, fea_after_fusion, x, fea_specific_vto, fea_vto, logits_vto
            gc.collect()

            return pred_vto
        elif task == "MSR":
            fea_specific_msp = self.block_specific_msr_feature_extractor(x)
            fea_msp = (fea_specific_msp + fea_after_fusion)
            fea_msp = self.msp_task_projection_head(fea_msp)
            fea_msp = fea_msp.view(fea_msp.size(0), -1)
            logits_msp = self.msp_task_classifier(fea_msp)
            pred_msp = F.softmax(logits_msp, dim = 1)

            del fea_shared_extract, fea_after_fusion, x, fea_specific_msp, fea_msp, logits_msp
            gc.collect()
            return pred_msp

def plot(confusion_matrix, classes):
    proportion = []
    length = len(confusion_matrix)
    print(length)
    for i in confusion_matrix:
        for j in i:
            temp = j / (np.sum(i))*100
            proportion.append(temp)
    # print(np.sum(confusion_matrix[0]))
    # print(proportion)
    pshow = []
    for i in proportion:
        pt = "%.2f" % (i)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(length, length)  # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(length, length)
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
    }
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
    # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
    # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
    # plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    thresh = confusion_matrix.max() / 2.
    iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (confusion_matrix.size, 2))
    for i, j in iters:
        if (i == j):
            # plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, color='white',
            #         weight=5)  # 显示对应的数字
            plt.text(j, i, pshow[i, j], va='center', ha='center', fontsize=10, color='white')
        else:
            # plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10)  # 显示对应的数字
            plt.text(j, i, pshow[i, j], va='center', ha='center', fontsize=10)

    # plt.ylabel('True label', fontsize=16)
    # plt.xlabel('Predict label', fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig("A.png", dpi=500)

# 生成MTR和MSR任务的混淆矩阵，分析ERP成分与各个脑区的对自监督任务的影响
def plot_generate_MTR_task_matrix(dataset_name):
    classes = ["C1", "vP1", "N1", "vN1", "P2", "vN2", "P3", "N3", "SW"]
    y_pred, y = get_data(dataset_name, "MTR")
    matrix = confusion_matrix(y, y_pred)
    plot(matrix, classes)

def plot_generate_MSR_task_matrix(dataset_name):
    classes = ["PF", "LT", "F", "RT", "LP", "C", "RP", "O"]
    y_pred, y = get_data(dataset_name, "MSR")
    matrix = confusion_matrix(y, y_pred)
    plot(matrix, classes)

def calculate_prediction_value(param, data, task):
    model = MTCN(n_class_primary=2).to(DEVICE)
    model.load_state_dict(param)
    model.eval()
    x = torch.tensor(data).to(torch.float32)
    x = x.to(DEVICE)
    y_pred = model(x, task)
    _, pred = torch.max(y_pred, 1)
    del x, y_pred, param, data
    gc.collect()
    return pred

def get_data(dataset_name, task_name):
    if dataset_name == "THU": sub_num = 64
    elif dataset_name == "CAS": sub_num = 14
    elif dataset_name == "GIST": sub_num = 55
    else: print("数据集名称输入错误")
    params_dir = os.path.join(os.getcwd(), "ExperimentForMTCN", f"{dataset_name}", "Params")
    data_dir = os.path.join(os.getcwd(), "Dataset", f"{dataset_name}")
    y_pred, y = [], []
    for i in range(1, sub_num+1):
        param = torch.load(os.path.join(params_dir, f"{dataset_name}_MTCN_{i:>02d}.pth"))
        if task_name == "MTR":
            data = np.load(os.path.join(data_dir, f"S{i:>02d}", "x_mtr_test.npy"))
            label = np.load(os.path.join(data_dir, f"S{i:>02d}", "y_mtr_test.npy"))
        elif task_name == "MSR":
            data = np.load(os.path.join(data_dir, f"S{i:>02d}", "x_msr_test.npy"))
            label = np.load(os.path.join(data_dir, f"S{i:>02d}", "y_msr_test.npy"))  
        mid_value = data.shape[0] // 2
        data_1, data_2 = data[: mid_value], data[mid_value: ]
        for data in [data_1, data_2]:
            pred = calculate_prediction_value(param, data, task_name)
            y_pred.extend(pred.cpu().detach())
        y.extend(label)
        print(len(y_pred), len(y))
        del data, label, param, pred
        gc.collect()
    y_pred, y = np.array(y_pred), np.array(y)
    return y_pred, y

if __name__ == "__main__":
    plot_generate_MTR_task_matrix("THU")
    plot_generate_MSR_task_matrix("THU")




