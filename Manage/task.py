'''
Author: Tammie li
Description: 定义各个模型的训练和测试任务（通用深度学习网络可以共用任务）
FilePath: \task.py
'''

import torch
import pickle
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import torch.nn as nn
import sklearn.decomposition
import sklearn.discriminant_analysis
import pywt
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
from Manage.loss import LossFunction


class FeatureQueue():
    def __init__(self, dim, length, device):
        self.length = length
        self.queue = torch.zeros(length, dim)
        self.ptr = 0
        self.device = device

    @torch.no_grad()
    def update(self, feat):
        batch_size = feat.shape[0]
        # assert self.length % batch_size == 0  # for simplicity

        # replace the features at ptr (dequeue and enqueue)
        self.queue[self.ptr:self.ptr + batch_size] = feat
        self.ptr = (self.ptr + batch_size) % self.length  # move pointer

    def get(self):
        tmp = self.queue.sum(axis=1)
        cnt = (tmp != 0).sum()
        res = torch.tensor(self.queue[:cnt, :]).to(self.device)
        return res


class OnlineFeatureAlignment:
    def __init__(self):
        pass
    def summarize(self, z):
        # feature summarization
        mu = z.mean(axis=0)
        sigma = self.covariance(z)
        return mu, sigma

    def covariance(self, features):
        assert len(features.size()) == 2, "TODO: multi-dimensional feature map covariance"
        n = features.shape[0]
        tmp = torch.ones((1, n), device=features.device) @ features
        cov = (features.t() @ features - (tmp.t() @ tmp) / n) / (n - 1)
        return cov

    def coral(self, cs, ct):
        d = cs.shape[0]
        loss = (cs - ct).pow(2).sum() / (4. * d ** 2)
        return loss

    def linear_mmd(self, ms, mt):
        loss = (ms - mt).pow(2).mean()
        return loss


class MOSATrainTask:
    def __init__(self, SubID, MODEL, DEVICE, dataset):
        self.sub_id = SubID
        self.device = DEVICE
        self.model = MODEL
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = LossFunction()
        self.dataset = dataset
    
    def _summarize(self, data):
        ttfa = OnlineFeatureAlignment()
        self.model.load_state_dict(torch.load(f'CheckPoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pth'))
        self.model.eval()
        with torch.no_grad():
            z_main, z_vto, z_msp = [], [], []
            for index, data in enumerate(data):
                x_train_main_task, y_train_main_task, x_train_vto_task, y_train_vto_task, x_train_msp_task, y_train_msp_task = data
                batch_size = x_train_main_task.shape[0] if index == 0 else batch_size
                x_train_main_task = x_train_main_task.type(torch.FloatTensor)
                x_train_vto_task = x_train_vto_task.type(torch.FloatTensor)
                x_train_msp_task = x_train_msp_task.type(torch.FloatTensor)

                x_train_main_task, y_train_main_task = x_train_main_task.to(self.device), y_train_main_task.to(self.device)
                x_train_vto_task, y_train_vto_task = x_train_vto_task.to(self.device), y_train_vto_task.to(self.device)
                x_train_msp_task, y_train_msp_task = x_train_msp_task.to(self.device), y_train_msp_task.to(self.device)

                x_train_vto_task = x_train_vto_task.reshape(x_train_vto_task.shape[0]*x_train_vto_task.shape[1], x_train_vto_task.shape[2], x_train_vto_task.shape[3])
                x_train_msp_task = x_train_msp_task.reshape(x_train_msp_task.shape[0]*x_train_msp_task.shape[1], x_train_msp_task.shape[2], x_train_msp_task.shape[3])
                y_train_vto_task = y_train_vto_task.reshape(y_train_vto_task.shape[0]*y_train_vto_task.shape[1])
                y_train_msp_task = y_train_msp_task.reshape(y_train_msp_task.shape[0]*y_train_msp_task.shape[1])

                fea = self.model(x_train_main_task, "GetFeature", "---")
                head_vto = self.model(x_train_vto_task, "GetHeadVTO", "---")
                head_msp = self.model(x_train_msp_task, "GetHeadMSP", "---")


                head_vto = head_vto.reshape(int(head_vto.shape[0]/9), 144)
                head_msp = head_msp.reshape(int(head_msp.shape[0]/8), 128)

                if(index == 0):
                    z_main, z_vto, z_msp = fea, head_vto, head_msp
                else:
                    try:
                        z_main = torch.cat((z_main, fea), 0)
                        z_vto = torch.cat((z_vto, head_vto), 0)
                        z_msp = torch.cat((z_msp, head_msp), 0)
                    except:
                        pass
        mu_main, sigma_main = ttfa.summarize(z_main)
        mu_vto, sigma_vto = ttfa.summarize(z_vto)
        mu_msp, sigma_msp = ttfa.summarize(z_msp)

        para_dict = {'mu_main': mu_main, 'sigma_main': sigma_main, 
                     'mu_vto': mu_vto, 'sigma_vto': sigma_vto,
                     'mu_msp': mu_msp, 'sigma_msp': sigma_msp}

        file = open(f'CheckPoint/{self.dataset}_TTFA_{self.sub_id:>02d}.pickle', 'wb')
        pickle.dump(para_dict, file)
        file.close()

    def train(self, dataloader, epochs):
        # print("************************************** MSOA Training Task ************************************")
        with tqdm(total=epochs, desc='Epoch', leave=True, ncols=100, unit_scale=True) as pbar:
            for epoch in range(epochs):
                self.model.train()
                running_loss = 0.0
                correct_num, correct_vto_num, correct_msp_num = 0, 0, 0
                sum_num = 0
                for index, data in enumerate(dataloader):
                    # generate new data and labels
                    x_train_main_task, y_train_main_task, x_train_vto_task, y_train_vto_task, x_train_msp_task, y_train_msp_task = data
                    batch_size = x_train_main_task.shape[0] if index == 0 else batch_size

                    sum_num += x_train_main_task.shape[0]
                    x_train_main_task = x_train_main_task.type(torch.FloatTensor)
                    x_train_vto_task = x_train_vto_task.type(torch.FloatTensor)
                    x_train_msp_task = x_train_msp_task.type(torch.FloatTensor)

                    x_train_main_task, y_train_main_task = x_train_main_task.to(self.device), y_train_main_task.to(self.device)
                    x_train_vto_task, y_train_vto_task = x_train_vto_task.to(self.device), y_train_vto_task.to(self.device)
                    x_train_msp_task, y_train_msp_task = x_train_msp_task.to(self.device), y_train_msp_task.to(self.device)

                    x_train_vto_task = x_train_vto_task.reshape(x_train_vto_task.shape[0]*x_train_vto_task.shape[1], x_train_vto_task.shape[2], x_train_vto_task.shape[3])
                    x_train_msp_task = x_train_msp_task.reshape(x_train_msp_task.shape[0]*x_train_msp_task.shape[1], x_train_msp_task.shape[2], x_train_msp_task.shape[3])
                    y_train_vto_task = y_train_vto_task.reshape(y_train_vto_task.shape[0]*y_train_vto_task.shape[1])
                    y_train_msp_task = y_train_msp_task.reshape(y_train_msp_task.shape[0]*y_train_msp_task.shape[1])

                    pred_primary = self.model(x_train_main_task, "trainStage", "main")
                    pred_vto = self.model(x_train_vto_task, "trainStage", "vto")
                    pred_msp = self.model(x_train_msp_task, "trainStage", "msp")

                    loss = self.criterion.calculateTrainStageLoss(pred_primary, y_train_main_task, pred_vto, y_train_vto_task,
                                                                pred_msp, y_train_msp_task)
                    _, pred = torch.max(pred_primary, 1)
                    _, pred_vto = torch.max(pred_vto, 1)
                    _, pred_msp = torch.max(pred_msp, 1)
                    
                    for i in range(y_train_main_task.shape[0]):
                        if y_train_main_task[i] == pred[i]:
                            correct_num += 1
                    for i in range(y_train_vto_task.shape[0]):
                        if y_train_vto_task[i] == pred_vto[i]:
                            correct_vto_num += 1
                    for i in range(y_train_msp_task.shape[0]):
                        if y_train_msp_task[i] == pred_msp[i]:
                            correct_msp_num += 1
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    running_loss += float(loss.item())

                _loss = running_loss / sum_num
                tmp_acc = correct_num / sum_num * 100
                tmp_acc_vto = correct_vto_num / sum_num / 9 * 100
                tmp_acc_msp = correct_msp_num / sum_num / 8 * 100

                
                # print(f'Train loss: {_loss:.4f}\tTrain acc: {acc:.2f}%')
                pbar.set_description(f'Epoch[{epoch}/{epochs}]')
                pbar.set_postfix(loss = _loss, acc = tmp_acc, acc_vto=tmp_acc_vto, acc_msp=tmp_acc_msp)
                pbar.update(1)
            path = f'Checkpoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pth'
            torch.save(self.model.state_dict(), path)

            # print('\n****************************Offline Summarization*****************************\n')
            self._summarize(dataloader)


class MOSATestTask:
    def __init__(self, SubID, MODEL, DEVICE, dataset):
        self.sub_id = SubID
        self.device = DEVICE
        self.model = MODEL
        self.dataset = dataset
        # self.queue_main = FeatureQueue(dim=2400, length=1024, device=DEVICE)
        # self.queue_vto = FeatureQueue(dim=144, length=576*2, device=DEVICE)
        # self.queue_msp = FeatureQueue(dim=128, length=1024, device=DEVICE)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = LossFunction()
        self.ttfa = OnlineFeatureAlignment()
        

    def test(self, dataloader, epochs):
        # print("************************************ MSOA Test Task ************************************")
        correct_num = 0
        sum_num = 0
        preds = []
        ys = []
        pred_score = []
        self.model.load_state_dict(torch.load(f'CheckPoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pth'))
        # 加载TTFA对齐参数
        with open(f'CheckPoint/{self.dataset}_TTFA_{self.sub_id:>02d}.pickle', 'rb') as file:
            ttfa_para = pickle.load(file)
        mu_main, sigma_main, mu_vto, sigma_vto, mu_msp, sigma_msp = ttfa_para['mu_main'], ttfa_para['sigma_main'], ttfa_para['mu_vto'], \
                                                                    ttfa_para['sigma_vto'], ttfa_para['mu_msp'], ttfa_para['sigma_msp'],
        mu_main, mu_vto, mu_msp = torch.tensor(mu_main).to(self.device), torch.tensor(mu_vto).to(self.device), torch.tensor(mu_msp).to(self.device)
        sigma_main, sigma_vto, sigma_msp = torch.tensor(sigma_main).to(self.device), torch.tensor(sigma_vto).to(self.device), torch.tensor(sigma_msp).to(self.device)
        for index, data in enumerate(dataloader):
            # 加载模型参数
            x_test_main_task, y_test_main_task, x_test_vto_task, y_test_vto_task, x_test_msp_task, y_test_msp_task = data
            x_test_main_task = x_test_main_task.type(torch.FloatTensor)
            x_test_vto_task = x_test_vto_task.type(torch.FloatTensor)
            x_test_msp_task = x_test_msp_task.type(torch.FloatTensor)

            sum_num = x_test_main_task.shape[0]

            x_test_main_task, y_test_main_task = x_test_main_task.to(self.device), y_test_main_task.to(self.device)
            x_test_vto_task, y_test_vto_task = x_test_vto_task.to(self.device), y_test_vto_task.to(self.device)
            x_test_msp_task, y_test_msp_task = x_test_msp_task.to(self.device), y_test_msp_task.to(self.device)

            x_test_vto_task = x_test_vto_task.reshape(x_test_vto_task.shape[0]*x_test_vto_task.shape[1], x_test_vto_task.shape[2], x_test_vto_task.shape[3])
            x_test_msp_task = x_test_msp_task.reshape(x_test_msp_task.shape[0]*x_test_msp_task.shape[1], x_test_msp_task.shape[2], x_test_msp_task.shape[3])
            y_test_vto_task = y_test_vto_task.reshape(y_test_vto_task.shape[0]*y_test_vto_task.shape[1])
            y_test_msp_task = y_test_msp_task.reshape(y_test_msp_task.shape[0]*y_test_msp_task.shape[1])
               
            self.model.train()
            with tqdm(total=epochs, desc='Epoch', leave=True, ncols=100, unit_scale=True) as pbar:
                for time in range(epochs):
                    # self.model.eval()
                    fea = self.model(x_test_main_task, "GetFeature", "---")
                    head_vto = self.model(x_test_vto_task, "GetHeadVTO", "---") 
                    head_msp = self.model(x_test_msp_task, "GetHeadMSP", "---")    

                    # self.queue_main.update(fea)
                    # self.queue_vto.update(head_vto.reshape(int(head_vto.shape[0]*head_vto.shape[1]/144), 144))
                    # self.queue_msp.update(head_msp.reshape(int(head_msp.shape[0]*head_msp.shape[1]/128), 128))

                    pred_vto = self.model(x_test_vto_task, "testStageI", "vto")
                    pred_msp = self.model(x_test_msp_task, "testStageI", "msp")
                                    
                    # loss_mean_main = self.ttfa.linear_mmd(self.queue_main.get().mean(axis=0), mu_main)
                    # loss_coral_main = self.ttfa.coral(self.ttfa.covariance(self.queue_main.get()), sigma_main)

                    # loss_mean_vto = self.ttfa.linear_mmd(self.queue_vto.get().mean(axis=0), mu_vto)
                    # loss_coral_vto = self.ttfa.coral(self.ttfa.covariance(self.queue_vto.get()), sigma_vto)

                    # loss_mean_msp = self.ttfa.linear_mmd(self.queue_msp.get().mean(axis=0), mu_msp)
                    # loss_coral_msp = self.ttfa.coral(self.ttfa.covariance(self.queue_msp.get()), sigma_msp)

                    loss = self.criterion.calculateTestStageILoss(pred_vto, y_test_vto_task, pred_msp, y_test_msp_task)
                    _, pred_vto = torch.max(pred_vto, 1)
                    _, pred_msp = torch.max(pred_msp, 1)

                    correct_vto_num, correct_msp_num = 0, 0
                    
                    for i in range(y_test_vto_task.shape[0]):
                        if y_test_vto_task[i] == pred_vto[i]:
                            correct_vto_num += 1
                    for i in range(y_test_msp_task.shape[0]):
                        if y_test_msp_task[i] == pred_msp[i]:
                            correct_msp_num += 1

                    # print(loss, loss_mean_main, loss_coral_main, loss_mean_vto, loss_coral_vto, loss_mean_msp, loss_coral_msp)

                    # loss =  loss + loss_mean_main + loss_coral_main + loss_mean_vto + loss_coral_vto + loss_mean_msp + loss_coral_msp
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    tmp_acc_vto = correct_vto_num / sum_num / 9 * 100
                    tmp_acc_msp = correct_msp_num / sum_num / 8 * 100

                    
                    pbar.set_description(f'Epoch[{time}/{epochs}]')
                    pbar.set_postfix(acc_vto=tmp_acc_vto, acc_msp=tmp_acc_msp)
                    pbar.update(1)
            self.model.eval()
            
            pred_primary = self.model(x_test_main_task, "testStageII", "main")
            _, pred = torch.max(pred_primary, 1)

            correct_num += np.sum(pred.cpu().numpy() == y_test_main_task.cpu().numpy())
            preds.extend(pred.cpu().numpy().tolist())
            ys.extend(y_test_main_task.cpu().tolist())    
            pred_score.extend(pred_primary.cpu().numpy().tolist())        
        acc = correct_num / sum_num * 100
        print(f'Test acc: {acc:.2f}%')

        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y', ys)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_preds', preds)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y_pred', pred_score)

        path = f'CheckPoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pth'
        torch.save(self.model.state_dict(), path)
        
        return acc


class GeneralTrainTask:
    def __init__(self, SubID, MODEL, DEVICE, dataset):
        self.sub_id = SubID
        self.device = DEVICE
        self.model = MODEL
        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())


    def train(self, dataloader, epochs):
        with tqdm(total=epochs, desc='Epoch', leave=True, ncols=100, unit_scale=True) as pbar:
            for epoch in range(epochs):
                # make dataset
                self.model.train()
                running_loss = 0.0
                correct_num = 0
                batch_size = None
                sum_num = 0
                for index, data in enumerate(dataloader):
                    x, y = data
                    batch_size = x.shape[0] if index == 0 else batch_size
                    x = torch.tensor(x).to(torch.float32)
                    y = torch.tensor(y).to(torch.long)  
                    x, y = x.to(self.device), y.to(self.device)           
                    y_pred = self.model(x)
                    loss = self.criterion(y_pred, y)
                    _, pred = torch.max(y_pred, 1)
                    correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    running_loss += float(loss.item())
                    sum_num += x.shape[0]
                batch_num = sum_num // batch_size
                _loss = running_loss / (batch_num + 1)
                acc = correct_num / sum_num * 100
                pbar.update(1)
                pbar.set_description(f'Epoch[{epoch}/{epochs}]')
                pbar.set_postfix(loss = _loss, acc = acc)
                # print(f'Train loss: {_loss:.4f}\tTrain acc: {acc:.2f}%') 

        path = f'Checkpoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pth'
        # path = f'Checkpoint/CrossSet/{self.dataset}_{self.model.__class__.__name__}.pth'


        
        torch.save(self.model.state_dict(), path)        


class GeneralTestTask:
    def __init__(self, SubID, MODEL, DEVICE, dataset):
        self.sub_id = SubID
        self.device = DEVICE
        self.model = MODEL
        self.dataset = dataset

    def test(self, data):
        self.model.load_state_dict(torch.load(f'Checkpoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pth'))
        # self.model.load_state_dict(torch.load(f'Checkpoint/CrossSet/CAS_{self.model.__class__.__name__}.pth'))
        running_loss = 0.0
        correct_num = 0
        self.model.eval()
        batch_size = None
        preds = []
        ys = []
        pred_score = []
        test_num = 0
        for index, data in enumerate(data):
            x, y = data
            batch_size = x.shape[0] if index == 0 else batch_size
            x = torch.tensor(x).to(torch.float32)
            y = torch.tensor(y).to(torch.long)  
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            _, pred = torch.max(y_pred, 1)

            correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
            preds.extend(pred.cpu().numpy().tolist())
            ys.extend(y.cpu().tolist())
            pred_score.extend(y_pred.cpu().detach().numpy())  
            test_num += x.shape[0]


        batch_num = test_num // batch_size
        acc = correct_num / test_num * 100
        print(f'Test acc: {acc:.2f}%')   
        
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y', ys)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_preds', preds)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y_pred', np.array(pred_score))


class RLDA:
    def __init__(self, model, dataset, sub_id) -> None:
        self.rlda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
            n_components=1)
        self.channel_num = 64
        self.model = model
        self.dataset = dataset
        self.sub_id = sub_id


    def wavelets(self, data):
        '''
        使用小波变换对数据进行特征提取
        data : 指定要处理的数据
        '''
        result = pywt.wavedec(data=data, wavelet="db4", level=5)
        result = np.array(result[0])
        return result[:, :200]

    def train(self, x: np, y: np) -> None:
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        self.rlda.fit(x, y)
        file = open(f'CheckPoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pickle', 'wb')
        pickle.dump(self.rlda, file)
        file.close()
        

    def test(self, x, y) -> np:
        with open(f'CheckPoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pickle', "rb") as file:
            self.rlda = pickle.load(file)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        # feature = self.wavelets(x)
        preds = self.rlda.predict(x)
        pred_y = self.rlda.predict_proba(x)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y', y)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_preds', preds)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y_pred', np.array(pred_y))


class XDAWN_RG:
    def __init__(self, model, dataset, sub_id) -> None:
        # N C T
        self.clf = make_pipeline(XdawnCovariances(estimator='oas'),
                                 TangentSpace(metric='riemann'),
                                 LogisticRegression())
        self.model = model
        self.dataset = dataset
        self.sub_id = sub_id



    def train(self, x: np, y: np) -> None:
        # x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        train_success_flag = False
        cnt = 0
        while((not train_success_flag) or cnt == 10):
            try:
                cnt += 1
                rd = np.random.permutation(x.shape[0])
                x = x[rd, :, :]
                y = y[rd]
                self.clf.fit(x, y)
                train_success_flag = True
            except np.linalg.LinAlgError:
                pass
        file = open(f'CheckPoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pickle', 'wb')
        pickle.dump(self.clf, file)
        file.close()

    def test(self, x, y) -> np:
        with open(f'CheckPoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pickle', "rb") as file:
            self.clf = pickle.load(file)
        # x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        preds = self.clf.predict(x)
        pred_y = self.clf.predict_proba(x)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y', y)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y_pred', pred_y)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_preds', preds)


class HDCA:
    def __init__(self, n, model, dataset, sub_id) -> None:
        self.n = n
        # self.rlda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)

        self.channel_num = 64
        self.model = model
        self.dataset = dataset
        self.sub_id = sub_id

    def _sub_mean(self, row_data):
        num_sample, num_chan, num_point = row_data.shape
        data = np.zeros((num_sample, num_chan, self.n))
        step = int(np.floor(num_point / self.n))
        # 假定一共有四段，对于前三段
        for i in range(self.n - 1):
            data[:, :, i] = np.mean(row_data[:, :, i * step:(i + 1) * step], axis=2)
        # 对于第四段
        data[:, :, self.n - 1] = np.mean(row_data[:, :, (self.n - 1) * step:], axis=2)
        return data

    def train(self, x: np, y: np) -> None:
        data = self._sub_mean(x)
        rlda_params = []
        preds = []
        for i in range(self.n):
            tmp_data = data[:, :, i]
            rlda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)
            rlda.fit(tmp_data, y)
            pred = rlda.predict(tmp_data)
            preds.append(pred)
            rlda_params.append(rlda)
        rlda_params = np.array(rlda_params)
        preds = np.array(preds).transpose()
        rg = LogisticRegression()
        rg.fit(preds, y)
        paras = dict({'stage1': rlda_params, 'stage2': rg})
        file = open(f'CheckPoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pickle', 'wb')
        pickle.dump(paras, file)
        file.close()

        

    def test(self, x, y) -> np:
        with open(f'CheckPoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pickle', 'rb') as file:
            params = pickle.load(file)

        data = self._sub_mean(x)

        para_rlda = params['stage1']
        para_lr = params['stage2']

        preds = []
        for i in range(self.n):
            tmp_data = data[:, :, i]
            pred = para_rlda[i].predict(tmp_data)
            preds.append(pred)
        preds = np.array(preds).transpose()
        
        fin_preds = para_lr.predict(preds)
        y_pred = para_lr.predict_proba(preds)

        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y', y)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_preds', fin_preds)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y_pred', np.array(y_pred))



class MTCNTrainTask:
    def __init__(self, SubID, MODEL, DEVICE, dataset):
        self.sub_id = SubID
        self.device = DEVICE
        self.model = MODEL
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = LossFunction()
        self.dataset = dataset


    def train(self, dataloader, epochs):
        with tqdm(total=epochs, desc='Epoch', leave=True, ncols=100, unit_scale=True) as pbar:
            for epoch in range(epochs):
                self.model.train()
                running_loss = 0.0
                correct_num, correct_vto_num, correct_msp_num = 0, 0, 0
                sum_num = 0
                for index, data in enumerate(dataloader):
                    # generate new data and labels

                    x_train_main_task, y_train_main_task, x_train_vto_task, y_train_vto_task, x_train_msp_task, y_train_msp_task = data
                    batch_size = x_train_main_task.shape[0] if index == 0 else batch_size

                    sum_num += x_train_main_task.shape[0]
                    x_train_main_task = x_train_main_task.type(torch.FloatTensor)
                    x_train_vto_task = x_train_vto_task.type(torch.FloatTensor)
                    x_train_msp_task = x_train_msp_task.type(torch.FloatTensor)

                    x_train_main_task, y_train_main_task = x_train_main_task.to(self.device), y_train_main_task.to(self.device)
                    x_train_vto_task, y_train_vto_task = x_train_vto_task.to(self.device), y_train_vto_task.to(self.device)
                    x_train_msp_task, y_train_msp_task = x_train_msp_task.to(self.device), y_train_msp_task.to(self.device)

                    x_train_vto_task = x_train_vto_task.reshape(x_train_vto_task.shape[0]*x_train_vto_task.shape[1], x_train_vto_task.shape[2], x_train_vto_task.shape[3])
                    x_train_msp_task = x_train_msp_task.reshape(x_train_msp_task.shape[0]*x_train_msp_task.shape[1], x_train_msp_task.shape[2], x_train_msp_task.shape[3])
                    y_train_vto_task = y_train_vto_task.reshape(y_train_vto_task.shape[0]*y_train_vto_task.shape[1])
                    y_train_msp_task = y_train_msp_task.reshape(y_train_msp_task.shape[0]*y_train_msp_task.shape[1])

                    pred_primary, loss_main = self.model(x_train_main_task, "main")
                    pred_vto, loss_vto = self.model(x_train_vto_task, "vto")
                    pred_msp, loss_msp = self.model(x_train_msp_task, "msp")

                    loss = self.criterion.calculateTrainStageLoss(pred_primary, y_train_main_task, pred_vto, y_train_vto_task,
                                                                pred_msp, y_train_msp_task)
                    
                    loss += loss_main + loss_vto + loss_msp

                    _, pred = torch.max(pred_primary, 1)
                    _, pred_vto = torch.max(pred_vto, 1)
                    _, pred_msp = torch.max(pred_msp, 1)
                    
                    for i in range(y_train_main_task.shape[0]):
                        if y_train_main_task[i] == pred[i]:
                            correct_num += 1
                    for i in range(y_train_vto_task.shape[0]):
                        if y_train_vto_task[i] == pred_vto[i]:
                            correct_vto_num += 1
                    for i in range(y_train_msp_task.shape[0]):
                        if y_train_msp_task[i] == pred_msp[i]:
                            correct_msp_num += 1
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    running_loss += float(loss.item())

                _loss = running_loss / sum_num
                tmp_acc = correct_num / sum_num * 100
                tmp_acc_vto = correct_vto_num / sum_num / 9 * 100
                tmp_acc_msp = correct_msp_num / sum_num / 8 * 100

                # print(f'Train loss: {_loss:.4f}\tTrain acc: {acc:.2f}%')
                pbar.set_description(f'Epoch[{epoch}/{epochs}]')
                pbar.set_postfix(loss = _loss, acc = tmp_acc, acc_vto=tmp_acc_vto, acc_msp=tmp_acc_msp)
                pbar.update(1)
            path = f'Checkpoint/CrossSet/{self.dataset}_{self.model.__class__.__name__}.pth'
            # path = f'Checkpoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pth'

            torch.save(self.model.state_dict(), path)


class MTCNTestTask:
    def __init__(self, SubID, MODEL, DEVICE, dataset):
        self.sub_id = SubID
        self.device = DEVICE
        self.model = MODEL
        self.dataset = dataset

    def test(self, data):
        # self.model.load_state_dict(torch.load(f'Checkpoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pth'))
        self.model.load_state_dict(torch.load(f'Checkpoint/CrossSet/CAS_{self.model.__class__.__name__}.pth'))
        running_loss = 0.0
        correct_num = 0
        self.model.eval()
        batch_size = None
        preds = []
        ys = []
        pred_score =[]
        test_num = 0
        # fea = []
        for index, data in enumerate(data):
            x, y = data
            batch_size = x.shape[0] if index == 0 else batch_size
            x = torch.tensor(x).to(torch.float32)
            y = torch.tensor(y).to(torch.long)  
            x, y = x.to(self.device), y.to(self.device)
            y_pred, loss = self.model(x, "main")
            _, pred = torch.max(y_pred, 1)

            correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
            preds.extend(pred.cpu().numpy().tolist())
            ys.extend(y.cpu().tolist())
            pred_score.extend(y_pred.cpu().detach().numpy())  
            test_num += x.shape[0]

        batch_num = test_num // batch_size
        acc = correct_num / test_num * 100
        print(f'Test acc: {acc:.2f}%')   

        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y', ys)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_preds', preds)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y_pred', np.array(pred_score))

class CPTrainTask:
    def __init__(self, SubID, MODEL, DEVICE, dataset):
        self.sub_id = SubID
        self.device = DEVICE
        self.model = MODEL
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.dataset = dataset
    
    def train(self, dataloader, epochs):
        print("Stage 1: representation learning")
        with tqdm(total=epochs, desc='Epoch', leave=True, ncols=100, unit_scale=True) as pbar:
            for mode in ["SSL", "SL"]: 
                for epoch in range(epochs):
                    self.model.train()
                    running_loss = 0.0
                    correct_num, correct_vto_num, correct_msp_num = 0, 0, 0
                    sum_num = 0
                    for index, data in enumerate(dataloader):
                        x, y = data
                        # 数据变换
                        if mode == "SSL":
                            value = x.shape[0] // 2
                            x[value: , :, :] = torch.cat((x[:value, :, :128], x[value: , :, 128:]), dim=2)
                            y[: value], y[value: ] = 1, 0
                        batch_size = x.shape[0] if index == 0 else batch_size
                        x = torch.tensor(x).to(torch.float32)
                        y = torch.tensor(y).to(torch.long)
                        x, y = x.to(self.device), y.to(self.device)   
                        if mode == "SSL":
                            y_pred = self.model(x, "feature")
                        else:
                            y_pred = self.model(x, "classifier")
                        loss = self.criterion(y_pred, y)
                        _, pred = torch.max(y_pred, 1)
                        correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        running_loss += float(loss.item())
                        sum_num += x.shape[0]
                    batch_num = sum_num // batch_size
                    _loss = running_loss / (batch_num + 1)
                    acc = correct_num / sum_num * 100
                    pbar.update(1)
                    pbar.set_description(f'Epoch[{epoch}/{epochs}]')
                    pbar.set_postfix(loss = _loss, acc = acc)
        path = f'Checkpoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pth'
        torch.save(self.model.state_dict(), path)        

class CPTestTask:
    def __init__(self, SubID, MODEL, DEVICE, dataset):
        self.sub_id = SubID
        self.device = DEVICE
        self.model = MODEL
        self.dataset = dataset

    def test(self, data):
        self.model.load_state_dict(torch.load(f'Checkpoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pth'))
        running_loss = 0.0
        correct_num = 0
        self.model.eval()
        batch_size = None
        preds = []
        ys = []
        pred_score = []
        test_num = 0
        # fea = []
        for index, data in enumerate(data):
            x, y = data
            batch_size = x.shape[0] if index == 0 else batch_size
            x = torch.tensor(x).to(torch.float32)
            y = torch.tensor(y).to(torch.long)  
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x, "test")
            _, pred = torch.max(y_pred, 1)

            correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
            preds.extend(pred.cpu().numpy().tolist())
            ys.extend(y.cpu().tolist())
            pred_score.extend(y_pred.cpu().detach().numpy())  
            test_num += x.shape[0]
        batch_num = test_num // batch_size
        acc = correct_num / test_num * 100
        print(f'Test acc: {acc:.2f}%')   
        
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y', ys)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_preds', preds)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y_pred', np.array(pred_score))

class TCTrainTask:
    def __init__(self, SubID, MODEL, DEVICE, dataset):
        self.sub_id = SubID
        self.device = DEVICE
        self.model = MODEL
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.dataset = dataset
    
    def train(self, dataloader, epochs):
        print("Stage 1: representation learning")
        with tqdm(total=epochs, desc='Epoch', leave=True, ncols=100, unit_scale=True) as pbar:
            for mode in ["SSL", "SL"]: 
                for epoch in range(epochs):
                    self.model.train()
                    running_loss = 0.0
                    correct_num, correct_vto_num, correct_msp_num = 0, 0, 0
                    sum_num = 0
                    for index, data in enumerate(dataloader):
                        x, y = data
                        # 数据变换
                        if mode == "SSL":
                            value = x.shape[0] // 2
                            x[value: , :, :] = torch.cat((x[:value, :, 162: ], x[:value, :, 81: 162], x[value: , :, :81]), dim=2)
                            y[: value], y[value: ] = 1, 0
                        batch_size = x.shape[0] if index == 0 else batch_size
                        x = torch.tensor(x).to(torch.float32)
                        y = torch.tensor(y).to(torch.long)
                        x, y = x.to(self.device), y.to(self.device)   
                        if mode == "SSL":
                            y_pred = self.model(x, "feature")
                        else:
                            y_pred = self.model(x, "classifier")
                        loss = self.criterion(y_pred, y)
                        _, pred = torch.max(y_pred, 1)
                        correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        running_loss += float(loss.item())
                        sum_num += x.shape[0]
                    batch_num = sum_num // batch_size
                    _loss = running_loss / (batch_num + 1)
                    acc = correct_num / sum_num * 100
                    pbar.update(1)
                    pbar.set_description(f'Epoch[{epoch}/{epochs}]')
                    pbar.set_postfix(loss = _loss, acc = acc)
        path = f'Checkpoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pth'
        torch.save(self.model.state_dict(), path)        

class TCTestTask:
    def __init__(self, SubID, MODEL, DEVICE, dataset):
        self.sub_id = SubID
        self.device = DEVICE
        self.model = MODEL
        self.dataset = dataset

    def test(self, data):
        self.model.load_state_dict(torch.load(f'Checkpoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pth'))
        running_loss = 0.0
        correct_num = 0
        self.model.eval()
        batch_size = None
        preds = []
        ys = []
        pred_score = []
        test_num = 0
        # fea = []
        for index, data in enumerate(data):
            x, y = data
            batch_size = x.shape[0] if index == 0 else batch_size
            x = torch.tensor(x).to(torch.float32)
            y = torch.tensor(y).to(torch.long)  
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x, "test")
            _, pred = torch.max(y_pred, 1)

            correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
            preds.extend(pred.cpu().numpy().tolist())
            ys.extend(y.cpu().tolist())
            pred_score.extend(y_pred.cpu().detach().numpy())  
            test_num += x.shape[0]
        batch_num = test_num // batch_size
        acc = correct_num / test_num * 100
        print(f'Test acc: {acc:.2f}%')   
        
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y', ys)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_preds', preds)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y_pred', np.array(pred_score))


class CPCTrainTask:
    def __init__(self, SubID, MODEL, DEVICE, dataset):
        self.sub_id = SubID
        self.device = DEVICE
        self.model = MODEL
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.dataset = dataset
    
    def train(self, dataloader, epochs):
        print("Stage 1: representation learning")
        with tqdm(total=epochs, desc='Epoch', leave=True, ncols=100, unit_scale=True) as pbar:
            for mode in ["SSL", "SL"]: 
                for epoch in range(epochs):
                    self.model.train()
                    running_loss = 0.0
                    correct_num, correct_vto_num, correct_msp_num = 0, 0, 0
                    sum_num = 0
                    for index, data in enumerate(dataloader):
                        x, y = data
                        # 数据变换
                        if mode == "SSL":
                            value = x.shape[0] // 2
                            tar_idx, notar_idx = torch.where(y == 0)[0], torch.where(y == 1)[0]
                            # 生成非目标-目标对
                            for i in range(min(len(notar_idx), len(tar_idx))):
                                tmp = torch.stack((x[tar_idx[i]], x[notar_idx[i]])).reshape(1, 2, 64, 256)
                                if i == 0: input = tmp
                                else: input = torch.cat((input, tmp))
                            tar_len, notar_len = len(tar_idx)-1, len(notar_idx)-1
                            # 生成目标-目标对
                            for i in range(value//2):
                                if tar_len > 7:
                                    tmp1 = x[tar_idx[i], ...]
                                    tmp2 = x[tar_idx[tar_len-i], ...]
                                    tmp = torch.stack((tmp1, tmp2)).reshape(1, 2, 64, 256)
                                    input = torch.cat((input, tmp))
                            # 生成非目标-非目标对
                            for i in range(value//2):
                                if notar_len > 7:
                                    tmp1 = x[notar_idx[i], ...]
                                    tmp2 = x[notar_idx[notar_len-i], ...]
                                    tmp = torch.stack((tmp1, tmp2)).reshape(1, 2, 64, 256)
                                    input = torch.cat((input, tmp))   
                            x = input                       
                            y = [0 for i in range(value)] + [1 for i in range(x.shape[0]-value)]
                        batch_size = x.shape[0] if index == 0 else batch_size
                        x = torch.tensor(x).to(torch.float32)
                        y = torch.tensor(y).to(torch.long)
                        x, y = x.to(self.device), y.to(self.device)   
                        if mode == "SSL":
                            y_pred = self.model(x, "feature")
                            loss = self.model.cal_loss(y_pred, y)
                        else:
                            y_pred = self.model(x, "classifier")
                            loss = self.criterion(y_pred, y)
                            _, pred = torch.max(y_pred, 1)
                            correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
                            sum_num += x.shape[0]
                            running_loss += float(loss.item())
                        self.optimizer.zero_grad()
                        if loss != 0: 
                            loss.backward()
                        self.optimizer.step() 
                    if mode == "SL":
                        batch_num = sum_num // batch_size
                        _loss = running_loss / (batch_num + 1)
                        acc = correct_num / sum_num * 100
                        pbar.update(1)
                        pbar.set_description(f'Epoch[{epoch}/{epochs}]')
                        pbar.set_postfix(loss = _loss, acc = acc)
        path = f'Checkpoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pth'
        torch.save(self.model.state_dict(), path)        

class CPCTestTask:
    def __init__(self, SubID, MODEL, DEVICE, dataset):
        self.sub_id = SubID
        self.device = DEVICE
        self.model = MODEL
        self.dataset = dataset

    def test(self, data):
        self.model.load_state_dict(torch.load(f'Checkpoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pth'))
        running_loss = 0.0
        correct_num = 0
        self.model.eval()
        batch_size = None
        preds = []
        ys = []
        pred_score = []
        test_num = 0
        # fea = []
        for index, data in enumerate(data):
            x, y = data
            batch_size = x.shape[0] if index == 0 else batch_size
            x = torch.tensor(x).to(torch.float32)
            y = torch.tensor(y).to(torch.long)  
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x, "test")
            _, pred = torch.max(y_pred, 1)

            correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
            preds.extend(pred.cpu().numpy().tolist())
            ys.extend(y.cpu().tolist())
            pred_score.extend(y_pred.cpu().detach().numpy())  
            test_num += x.shape[0]
        batch_num = test_num // batch_size
        acc = correct_num / test_num * 100
        print(f'Test acc: {acc:.2f}%')   
        
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y', ys)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_preds', preds)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y_pred', np.array(pred_score))


                



class TaskManage:
    def __init__(self, SubjectID, Name, Type, Epoch, Model, Data, DEVICE, DatasetName):
        self.model_name = Name
        self.task_type = Type
        self.epoch = Epoch
        self.model = Model
        self.data = Data
        self.device = DEVICE
        self.sub_id = SubjectID
        self.dataset_name = DatasetName

    def goTask(self):
        if self.model_name == 'MSOA':
            if(self.task_type == True):
                msoa = MOSATrainTask(self.sub_id, self.model, self.device, self.dataset_name)
                msoa.train(self.data, self.epoch)
                # print("\n ***************************************Training Finish!***********************************\n")
            else:
                msoa = MOSATestTask(self.sub_id, self.model, self.device, self.dataset_name)
                # msoa.test(self.data, self.epoch)
                # print("\n *****************************************Test Finish!***********************************\n")
        if self.model_name == 'PLNet' or self.model_name == 'EEGNet' or self.model_name == 'EEGInception' or \
            self.model_name == 'DeepConvNet' or self.model_name == 'PPNN' or self.model_name == 'DRL' or self.model_name == "PhyTransformer"  or \
                self.model_name == "Transformer" or self.model_name == "Informer" or self.model_name == "ConvTransformer" or self.model_name == "PhyTransformer_T" \
                    or self.model_name == "PhyTransformer_S" or self.model_name == "PhyTransformer_C" or self.model_name == "PhyTransformer_F":
            # 通用模型训练
            if(self.task_type == True):
                general = GeneralTrainTask(self.sub_id, self.model, self.device, self.dataset_name)
                general.train(self.data, self.epoch)
            else:
                general = GeneralTestTask(self.sub_id, self.model, self.device, self.dataset_name)
                general.test(self.data)
        if self.model_name == "MTCN":
            if(self.task_type == True):
                general = MTCNTrainTask(self.sub_id, self.model, self.device, self.dataset_name)
                general.train(self.data, self.epoch)
            else:
                general = MTCNTestTask(self.sub_id, self.model, self.device, self.dataset_name)
                general.test(self.data)
        if self.model_name == "CP":
            if(self.task_type == True):
                general = CPTrainTask(self.sub_id, self.model, self.device, self.dataset_name)
                general.train(self.data, self.epoch)
            else:
                general = CPTestTask(self.sub_id, self.model, self.device, self.dataset_name)
                general.test(self.data)
        if self.model_name == "TC":
            if(self.task_type == True):
                general = TCTrainTask(self.sub_id, self.model, self.device, self.dataset_name)
                general.train(self.data, self.epoch)
            else:
                general = TCTestTask(self.sub_id, self.model, self.device, self.dataset_name)
                general.test(self.data)
        if self.model_name == "CPC":
            if(self.task_type == True):
                general = CPCTrainTask(self.sub_id, self.model, self.device, self.dataset_name)
                general.train(self.data, self.epoch)
            else:
                general = CPCTestTask(self.sub_id, self.model, self.device, self.dataset_name)
                general.test(self.data)
        if self.model_name == 'rLDA':
            rLDA = RLDA(self.model, self.dataset_name, self.sub_id)
            if(self.task_type == True):
                rLDA.train(self.data['x'], self.data['y'])
            else:
                rLDA.test(self.data['x'], self.data['y'])

        if self.model_name == 'HDCA':
            hdca = HDCA(20, self.model, self.dataset_name, self.sub_id)
            if(self.task_type == True):
                hdca.train(self.data['x'], self.data['y'])
            else:
                hdca.test(self.data['x'], self.data['y'])

        if self.model_name == 'xDAWNRG':
            clf =XDAWN_RG(self.model, self.dataset_name, self.sub_id)
            if(self.task_type == True):
                clf.train(self.data['x'], self.data['y'])
            else:
                clf.test(self.data['x'], self.data['y'])




