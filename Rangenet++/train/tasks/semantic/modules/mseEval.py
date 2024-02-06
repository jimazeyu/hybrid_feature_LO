#!/usr/bin/env python3

import torch
import numpy as np

class mseEval:
    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):
        self.sum_squared_error = 0.0
        self.total_samples = 0

    def addBatch(self, predictions, targets):
        # 确保预测和目标都在正确的设备上
        if isinstance(predictions, np.ndarray):
            predictions = torch.from_numpy(predictions).float().to(self.device)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float().to(self.device)

        # 确保预测和目标为float类型以计算MSE
        predictions = predictions.float()
        targets = targets.float()

        # 计算平方差
        squared_error = (predictions - targets) ** 2

        # 更新总平方误差和样本数量
        self.sum_squared_error += torch.sum(squared_error).item()
        self.total_samples += predictions.numel()

    def getMSE(self):
        # 计算并返回MSE
        if self.total_samples == 0:
            raise RuntimeError("No samples were added to the evaluator.")
        mse = self.sum_squared_error / self.total_samples
        return mse
