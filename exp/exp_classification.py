from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pywt
import pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)#0.483
        model_optim = torch.optim.NAdam(self.model.parameters(), lr=self.args.learning_rate)  # 0.5352
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def _disentangle(self, x, w, j, save_path=None):
        """
        对输入的torch张量进行小波分解，并可视化和保存分解结果。

        参数:
        - x: torch.Tensor，输入数据，假设形状为 [16, 25246, 5]
        - w: str，小波基函数名称
        - j: int，小波分解层数
        - save_path: str，保存图片的路径，默认为 None，不保存图片

        返回:
        - xl: torch.Tensor，低频部分的重构结果
        - xh: torch.Tensor，高频部分的重构结果
        """
        # print(j)
        # 将 torch 张量转换为 numpy 数组并转置数据，使样本长度维度位于最后
        x_np = np.transpose(x.numpy(), (0, 2, 1))  # [16, 5, 25246]

        # 对样本长度维度进行小波分解
        # coef = pywt.wavedec(x_np, w, level=j, axis=-1)
        # 对样本长度维度进行小波分解
        coef = pywt.wavedec(x_np, w, level=j, axis=-1, mode='per')

        # 分离低频和高频系数
        coefl = [coef[0]]
        for i in range(len(coef) - 1):
            coefl.append(None)

        coefh = [None]
        for i in range(len(coef) - 1):
            coefh.append(coef[i + 1])

        # 小波重构
        # xl_np = pywt.waverec(coefl, w, axis=-1).transpose(0, 2, 1)  # [16, 25246, 5]
        # xh_np = pywt.waverec(coefh, w, axis=-1).transpose(0, 2, 1)  # [16, 25246, 5]
        # 小波重构，设置 mode='per' 来减少边界效应
        xl_np = pywt.waverec(coefl, w, axis=-1, mode='per')  # [16, 25246, 5]
        xh_np = pywt.waverec(coefh, w, axis=-1, mode='per')  # [16, 25246, 5]
        # 修正重构后的序列长度与原始长度不一致的问题
        xl_np = xl_np[..., :x.shape[1]]
        xh_np = xh_np[..., :x.shape[1]]
        xl_np=xl_np.transpose(0, 2, 1)
        xh_np = xh_np.transpose(0, 2, 1)

        # 将 numpy 数组转换回 torch 张量
        xl = torch.from_numpy(xl_np)
        xh = torch.from_numpy(xh_np)


        # 可视化和保存图片
        if save_path is not None:
            batch_size, seq_len, num_channels = x.shape
            for i in range(batch_size):
                plt.figure(figsize=(14, 10))
                for ch in range(num_channels):
                    # 原始信号
                    plt.subplot(num_channels, 3, 3 * ch + 1)
                    plt.plot(x[i, :, ch].numpy(), label='Original', color='blue')
                    plt.title(f'Sample {i + 1} Channel {ch + 1} - Original')
                    plt.legend()

                    # 低频信号
                    plt.subplot(num_channels, 3, 3 * ch + 2)
                    plt.plot(xl[i, :, ch].numpy(), label='Low Frequencies', color='green')
                    plt.title(f'Sample {i + 1} Channel {ch + 1} - Low Frequencies')
                    plt.legend()

                    # 高频信号
                    plt.subplot(num_channels, 3, 3 * ch + 3)
                    plt.plot(xh[i, :, ch].numpy(), label='High Frequencies', color='red')
                    plt.title(f'Sample {i + 1} Channel {ch + 1} - High Frequencies')
                    plt.legend()
                # 创建保存目录（如果不存在）
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                plt.tight_layout()
                plt.savefig(f'{save_path}/wavelet_decomposition_sample_{i + 1}.png')
                plt.close()
        return xl, xh

    def plot_curves(self,train_losses, val_losses, train_accuracies, val_accuracies):
        epochs = len(train_losses)
        plt.figure(figsize=(12, 6))

        # Plot loss curves
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()

        # Plot accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
        plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()

        plt.tight_layout()
        plt.savefig('loss_accuracy_curves.png')
        plt.close()

    def calculate_accuracy(self,outputs, labels):
        preds = torch.argmax(torch.nn.functional.softmax(outputs, dim=1), dim=1).cpu().numpy()
        trues = labels.flatten().cpu().numpy()
        return np.mean(preds == trues)

    def plot_confusion_matrix(self, labels, preds,num):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('train_confusion_matrix.png')
        plt.show()

        # 计算并打印每个类别的准确率
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        print(self.args.root_path)
        print(self.args.sensornum)
        with open('Resensor.txt', 'a') as f:
            f.write(f"{self.args.root_path}\n")
            f.write(f"传感器个数{self.args.sensornum}\n")
            for i, acc in enumerate(class_accuracy):
                f.write(f'level{num}类别 {i} 的准确率: {acc:.4f}\n')
        f.close()
        for i, acc in enumerate(class_accuracy):
            print(f'level{num}类别 {i} 的准确率: {acc:.4f}')

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask, filenames) in enumerate(vali_loader):
                batch_xl, batch_xh = self._disentangle(batch_x, 'sym2', self.args.level)  # sym2

                batch_xl = batch_xl.float().to(self.device)
                batch_xh = batch_xh.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_xl, batch_xh,padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze(-1).cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        all_preds = []
        all_labels = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_accuracy = []

            self.model.train()
            epoch_time = time.time()

            correct = 0
            total = 0

            for i, (batch_x, label, padding_mask, filenames) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                # print('batch_x',batch_x.shape)
                # batch_xl, batch_xh = self._disentangle(batch_x, 'sym2', 2)  # sym2
                # batch_xl, batch_xh = self._disentangle(batch_x, 'sym2', 2, save_path='/data/LJ/dataset')
                batch_xl, batch_xh = self._disentangle(batch_x, 'sym2', self.args.level, save_path=None)
                # print('batch_xl', batch_xl.shape)
                # print('batch_xh', batch_xh.shape)
                batch_xl = batch_xl.float().to(self.device)
                batch_xh = batch_xh.float().to(self.device)
                # print('batch_xl', batch_xl.shape)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_xl, batch_xh,padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                accuracy = self.calculate_accuracy(outputs, label)
                train_accuracy.append(accuracy)

                ##0717
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                ##################

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_accuracy = np.average(train_accuracy)

            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(vali_loss)
            val_accuracies.append(val_accuracy)

            # print(
            #     "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
            #     .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # Plot loss and accuracy curves
        self.plot_curves(train_losses, val_losses, train_accuracies, val_accuracies)
        # 计算并绘制混淆矩阵
        self.plot_confusion_matrix(all_labels, all_preds,self.args.level)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        combined_list=[]
        correct_files = []
        incorrect_files = []
        error_count_per_class = defaultdict(int)  # 用于统计每个类别的错误个数
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask, filenames) in enumerate(test_loader):
                batch_xl, batch_xh = self._disentangle(batch_x, 'sym2', self.args.level)  # sym2

                batch_xl = batch_xl.float().to(self.device)
                batch_xh = batch_xh.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_xl, batch_xh,padding_mask, None, None)
                # print(filenames)
                preds.append(outputs.detach())
                trues.append(label)

                # 计算当前批次的预测
                probs = torch.nn.functional.softmax(outputs, dim=1)
                predictions1 = torch.argmax(probs, dim=1).cpu().numpy()
                combined_list+=filenames

                # 将文件名根据预测结果分到正确或错误的列表中
                for j in range(len(predictions1)):
                    true_label = label.cpu().numpy()[j]
                    predicted_label = predictions1[j]
                    if predicted_label == true_label:
                        correct_files.append(filenames[j])
                    else:
                        incorrect_files.append(f"{filenames[j]} (predicted as {predicted_label})")
                        error_count_per_class[predicted_label] += 1  # 增加预测错误类别的计数

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        # filenameall = torch.cat(filenameall, 0)
        # print('test shape:', preds.shape, trues.shape)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # Confusion matrix and classification report
        cm = confusion_matrix(trues, predictions)
        report = classification_report(trues, predictions, target_names=[str(i) for i in range(len(np.unique(trues)))])

        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        print(" | ".join([f"class {idx}: {accuracy1:.4f}" for idx, accuracy1 in enumerate(class_accuracy)]))
        with open('Resensor.txt', 'a') as f:
            f.write(f'{" | ".join([f"class {idx}: {accuracy1:.4f}" for idx, accuracy1 in enumerate(class_accuracy)])}\n')
            f.write('accuracy: {}\n'.format(accuracy))
        f.close()
        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[str(i) for i in range(len(np.unique(trues)))],
                    yticklabels=[str(i) for i in range(len(np.unique(trues)))])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(folder_path, 'testconfusion_matrix.png'))
        plt.close()

        # Save the file names for correct and incorrect classifications
        with open('correct_files.txt', 'w') as f:
            for file in correct_files:
                f.write(f"{file}\n")

        with open('incorrect_files.txt', 'w') as f:
            for file in incorrect_files:
                f.write(f"{file}\n")

        # Save the error counts per class
        with open('error_count_per_class.txt', 'w') as f:
            for class_id, count in error_count_per_class.items():
                f.write(f"Class {class_id}: {count} errors\n")

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return
