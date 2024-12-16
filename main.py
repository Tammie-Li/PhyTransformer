import json
import torch

from Manage.data import DataManage
from Manage.task import TaskManage
from Manage.model import *
from Manage.evaluate import *
from Utils.record import *
import warnings
import gc
import mne
warnings.filterwarnings("ignore")

CONTROL = [True, False]  # 数据集控制位，True表示计算该数据集，False反之 （用于多服务器训练）
NUMCLASSES = 2
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE: ', DEVICE)

ModelName =  ["EEGNet", "DeepConvNet", "EEGInception", "PLNet", "HDCA", "xDAWNRG", "rLDA", "PPNN", "DRL", "CP", "TC", "CPC", "MSOA", "PhyTransformer", "Transformer", "Informer", "ConvTransformer", 
              "PhyTransformer_T", "PhyTransformer_S", "PhyTransformer_C", "PhyTransformer_F"]
ModelType = [True, True, True, True, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
Model = [EEGNet(NUMCLASSES), DeepConvNet(NUMCLASSES), EEGInception(NUMCLASSES), PLNet(NUMCLASSES), HDCA(NUMCLASSES), xDAWNRG(NUMCLASSES), 
         rLDA(NUMCLASSES), MTCN(NUMCLASSES), PPNN(NUMCLASSES), DRL(NUMCLASSES), CP(), TC(), CPC(), MSOA(NUMCLASSES), \
            PhyTransformer(num_classes=2, input_size=(1, 64, 256), sampling_rate=256, num_T=8, num_S=16, hidden=32, dropout_rate=0.5), \
                Transformer(64, NUMCLASSES), Informer(256, NUMCLASSES, 100), ConvTransformer(),\
                PhyTransformer_T(num_classes=2, input_size=(1, 64, 256), sampling_rate=256, num_T=8, num_S=16, hidden=32, dropout_rate=0.5),\
                PhyTransformer_S(num_classes=2, input_size=(1, 64, 256), sampling_rate=256, num_T=8, num_S=16, hidden=32, dropout_rate=0.5),\
                PhyTransformer_C(num_classes=2, input_size=(1, 64, 256), sampling_rate=256, num_T=8, num_S=16, hidden=32, dropout_rate=0.5),\
                PhyTransformer_F(num_classes=2, input_size=(1, 64, 256), sampling_rate=256, num_T=8, num_S=16, hidden=32, dropout_rate=0.5)]
if __name__ == '__main__':
    # 加载工程参数
    with open('config.json', 'r') as f:
        proj_config_para = json.load(f)

    _dataset_name = proj_config_para['DatasetName']

    sub_num = [proj_config_para['Dataset'][_dataset_name[i]]['subject_num'] for i in range(len(_dataset_name))]

    for i in range(1):
        for dataset_num in range(len(sub_num)):
            # 数据集层
            if CONTROL[dataset_num] is False: continue
            for subject_id in range(1, sub_num[dataset_num]+1, 1):
                for idx in range(1, len(ModelType)):
                # 数据集中的被试层
                # 完成数据加载操作
                # for idx in range(len(ModelName)-1):
                    # 对于不同模型
                    model = Model[idx].to(DEVICE)
                    model = model.to(DEVICE)
                    print(proj_config_para['DatasetName'][dataset_num], "\t", f'Subject_S{subject_id:>02d}', "\t", ModelName[idx])
                    
                    # 训练阶段
                    TrainDataManager = DataManage(Name = ModelName[idx], DataName = _dataset_name[dataset_num], Mode = True, 
                                                SubID = subject_id, BatchSize=proj_config_para['TrainPara']['batch_size'])
                    train_data_torch, train_x_npy, train_y_npy = TrainDataManager.getData()
                    train_data_trad = dict({'x': train_x_npy, 'y': train_y_npy})
                    print(train_x_npy.shape, train_y_npy.shape)
                    if ModelType[idx] is False:
                        # 传统方法处理
                        task_train_manage = TaskManage(subject_id, ModelName[idx], True, proj_config_para['TrainPara']['epoch'], 
                            model, train_data_trad, DEVICE, _dataset_name[dataset_num])
                        task_train_manage.goTask()

                    else:
                        # 深度学习算法处理
                        task_train_manage = TaskManage(subject_id, ModelName[idx], True, proj_config_para['TrainPara']['epoch'], 
                                                model, train_data_torch, DEVICE, _dataset_name[dataset_num])
                        task_train_manage.goTask()

                    # 测试阶段
                    TestDataManage = DataManage(Name = ModelName[idx], DataName = _dataset_name[dataset_num], Mode = False, 
                                                SubID = subject_id, BatchSize=proj_config_para['TestPara']['batch_size'])
                    test_data_torch, test_x_npy, test_y_npy = TestDataManage.getData()
                    test_data_trad = dict({'x': test_x_npy, 'y': test_y_npy})
                    if ModelType[idx] is False:
                        # 传统方法处理
                        task_train_manage = TaskManage(subject_id, ModelName[idx], False, proj_config_para['TrainPara']['epoch'], 
                            model, test_data_trad, DEVICE, _dataset_name[dataset_num])
                        task_train_manage.goTask()
                    else:
                        # 深度学习算法处理
                        task_test_manage = TaskManage(subject_id, ModelName[idx], False, proj_config_para['TestPara']['epoch'], 
                                                model, test_data_torch, DEVICE, _dataset_name[dataset_num])
                        task_test_manage.goTask()
                    
                    evaluateManager = EvaluateManage(subject_id, _dataset_name[dataset_num], ModelName[idx])

                    score = evaluateManager.calculate_metric_score()

                    # print("AUC: ", auc)

                    # 结果以及模型保存
                    update_result(score, _dataset_name[dataset_num], ModelName[idx], subject_id, ModelType[idx])
                
                del test_x_npy, test_y_npy, test_data_torch

                gc.collect()


                    
                