# Author: Tammie li
# Description: 作图 体现预测情况
# FilePath: \Utils\preprocess.py

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_prediction_curve(dataset_name, model_name, subject_id): 
    pred = np.load(f'PredictionResult/{dataset_name}_{model_name}_S{subject_id:>02d}_preds.npy')
    y = np.load(f'PredictionResult/{dataset_name}_{model_name}_S{subject_id:>02d}_y.npy')
    # 计算
    result = []
    for idx, label in enumerate(y):
        if pred[idx] == label:
            result.append(1)
        else:
            result.append(0)
    x = [i for i in range(len(result))]
    plt.plot(x, result, linewidth=2)
    plt.show()


if __name__ == "__main__":
    dataset_name = "CAS"
    model_name = "EEGNet"
    subject_id = 1

    plot_prediction_curve(dataset_name, model_name, subject_id)