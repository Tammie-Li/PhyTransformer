    # def plot_ROC_curve(self):
    #     fpr, tpr, thresholds = roc_curve(self.y, self.y_pred[:, 1])
    #     self.ROC_lib.append([fpr, tpr])
    #     plt.plot(fpr, tpr, linewidth=2, label=label)
    #     plt.show()

    # def plot_confusion_matrix(self, data):
        # 使用数据集中所有被试的平均值绘制混淆矩阵
        # data->[subject_num, 4]
        # TP_avg = sum(data[:, 0]) / data.shape[0]
        # TN_avg = sum(data[:, 1]) / data.shape[0]
        # FP_avg = sum(data[:, 2]) / data.shape[0]
        # FN_avg = sum(data[:, 3]) / data.shape[0]
        # pass