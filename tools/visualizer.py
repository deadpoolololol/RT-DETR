import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

class Visualizer:
    def __init__(self, save_dir=None):
        plt.ion()
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 5))
        self.train_loss_values = []
        self.eval_mAP_values = []
        # 保存指标的文件路径
        self.save_dir = save_dir
        self.train_loss_file = os.path.join(self.save_dir, 'train_loss_history.csv') 
        self.eval_mAP_file = os.path.join(self.save_dir, 'eval_mAP_history.csv') 
        # 恢复训练时，加载历史数据
        self.train_loss_history, self.eval_mAP_history = self.load_history()
        # 如果有历史数据，可以继续绘制
        if not self.train_loss_history.empty:
            self.train_loss_values = self.train_loss_history['train_loss'].tolist()
        if not self.eval_mAP_history.empty:
            self.eval_mAP_values = self.eval_mAP_history['eval_mAP'].tolist()


    def update_train_plot(self, train_loss, epoch,save=False):
        self.train_loss_values.append(train_loss.cpu().item())
        self.axs[0].clear()
        self.axs[0].plot(self.train_loss_values, label='Training Loss')
        self.axs[0].set_title("Training Loss")
        self.axs[0].set_xlabel("Iterations")
        self.axs[0].set_ylabel("Loss")
        self.axs[0].legend()
        plt.pause(0.01)

        # 保存图像
        if save and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            plt.savefig(os.path.join(self.save_dir, f"train_loss_epoch_{epoch}.png"))

            # 保存训练损失到文件
        if os.path.exists(self.train_loss_file):
            # 如果文件存在，则加载并追加新的数据
            history = pd.read_csv(self.train_loss_file)
            # 检查是否已有该 epoch 数据
            if epoch not in history['epoch'].values:
                new_row = pd.DataFrame([{'epoch': epoch, 'train_loss': train_loss.cpu().item()}])
                history = pd.concat([history, new_row], ignore_index=True)
        else:
            # 如果文件不存在，则创建新的文件并保存
            history = pd.DataFrame({'epoch': [epoch], 'train_loss': [train_loss.cpu().item()]})

        history.to_csv(self.train_loss_file, index=False)

    def update_eval_plot(self, eval_mAP, epoch,save=True):
        self.eval_mAP_values.append(eval_mAP)
        self.axs[1].clear()
        self.axs[1].plot(self.eval_mAP_values, label='Evaluation mAP', color='orange')
        self.axs[1].set_title("Evaluation mAP")
        self.axs[1].set_xlabel("Epochs")
        self.axs[1].set_ylabel("mAP")
        self.axs[1].legend()
        plt.pause(0.01)

        # 保存图像
        if save and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            # plt.savefig(os.path.join(save_dir, f"eval_mAP_epoch_{epoch}.png"))
            plt.savefig(os.path.join(self.save_dir, f"eval_mAP_epoch.png"))

         # 保存评估 mAP 到文件
        if os.path.exists(self.eval_mAP_file):
            # 如果文件存在，则加载并追加新的数据
            history = pd.read_csv(self.eval_mAP_file)
            # 检查是否已有该 epoch 数据
            if epoch not in history['epoch'].values:
                new_row = pd.DataFrame([{'epoch': epoch, 'eval_mAP': eval_mAP}])
                history = pd.concat([history, new_row], ignore_index=True)
        else:
            # 如果文件不存在，则创建新的文件并保存
            history = pd.DataFrame({'epoch': [epoch], 'eval_mAP': [eval_mAP]})

        history.to_csv(self.eval_mAP_file, index=False)

    def load_history(self):
        """
        加载之前保存的训练损失和评估指标数据
        :return: 返回训练损失和评估指标的历史数据
        """
        if os.path.exists(self.train_loss_file):
            train_loss_history = pd.read_csv(self.train_loss_file)
        else:
            train_loss_history = pd.DataFrame({'epoch': [], 'train_loss': []})

        if os.path.exists(self.eval_mAP_file):
            eval_mAP_history = pd.read_csv(self.eval_mAP_file)
        else:
            eval_mAP_history = pd.DataFrame({'epoch': [], 'eval_mAP': []})

        return train_loss_history, eval_mAP_history
    
if __name__ == '__main__':
    vis = Visualizer(save_dir="output")
    vis.update_train_plot(train_loss=torch.tensor(0.5), epoch=1, save=True)
    vis.update_train_plot(train_loss=torch.tensor(0.4), epoch=1, save=True)  # 不会重复添加
    vis.update_train_plot(train_loss=torch.tensor(0.3), epoch=2, save=True)  # 会添加新行