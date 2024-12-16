# Author: Tammie li
# Description: 作图 体现分布偏移问题
# FilePath: \Utils\preprocess.py

import numpy as np
from sklearn.manifold import TSNE
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

SUBID = 1

def t_SNE_transfer(data):
    data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    tsne = TSNE(n_components=2)
    tsne.fit_transform(data)

    print("stage 2: data_transfer_success!")
    return tsne.embedding_

# def construct_data_frame(data, label):
def construct_data_frame(data1, data2, data3, data4):
    data = np.concatenate((data1, data2, data3, data4))
    tb_feature = pd.DataFrame(data, columns=['x1','x2'])
    label = np.array(['tar_train' for i in range(data1.shape[0])] +
                     ['notar_train' for i in range(data2.shape[0])] +
                     ['tar_test' for i in range(data3.shape[0])] +
                     ['notar_test' for i in range(data4.shape[0])])
    tb_feature['class'] = np.array(label)

    return tb_feature


def plot_t_SNE(data_frame, subject_id):
    # sns.set_theme(style='white', font='Times New Roman', font_scale=1.3)
    color=['red','green','black','blue','black']
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
    plt.show()
    scatter_fig = g.get_figure()
    # scatter_fig.savefig(f'image/raw/S{subject_id:>02d}_raw.jpg', dpi = 600, bbox_inches='tight')


if __name__ == "__main__":
    x_train = np.load(os.path.join(os.getcwd(), 'Dataset', 'CAS', f'S{SUBID:>02d}', f'x_train.npy'))
    y_train = np.load(os.path.join(os.getcwd(), 'Dataset', 'CAS', f'S{SUBID:>02d}', f'y_train.npy'))
    x_test = np.load(os.path.join(os.getcwd(), 'Dataset', 'CAS', f'S{SUBID:>02d}', f'x_test.npy'))
    y_test = np.load(os.path.join(os.getcwd(), 'Dataset', 'CAS', f'S{SUBID:>02d}', f'y_test.npy'))

    tar_train, notar_train = np.where(y_train == 1)[0], np.where(y_train == 0)[0]
    tar_test, notar_test = np.where(y_test == 1)[0], np.where(y_test == 0)[0]

    x_tar_train, x_notar_train = x_train[tar_train], x_train[notar_train]
    x_tar_test, x_notar_test = x_test[tar_train], x_test[notar_train]

    # 降采样处理
    rand = np.random.permutation(x_notar_test.shape[0])
    x_notar_test = x_notar_test[rand[:x_tar_test.shape[0]]]

    # 提特征
    
    x_tar_train = t_SNE_transfer(x_tar_train)
    x_notar_train = t_SNE_transfer(x_notar_train)
    x_tar_test = t_SNE_transfer(x_tar_test)
    x_notar_test = t_SNE_transfer(x_notar_test)

    print(x_tar_train.shape, x_notar_train.shape, x_tar_test.shape, x_notar_test.shape)

    fea = construct_data_frame(x_tar_train, x_notar_train, x_tar_test, x_notar_test)

    plot_t_SNE(fea, SUBID)





