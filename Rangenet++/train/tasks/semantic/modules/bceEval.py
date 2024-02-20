#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np

class bceEval:
    def __init__(self, device):
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss().to(device)
        self.total_loss = 0.0
        self.count = 0

    def reset(self):
        self.total_loss = 0.0
        self.count = 0

    def addBatch(self, predictions, targets):
        predictions = predictions.to(self.device)
        targets = targets.to(self.device)
        loss = self.criterion(predictions, targets)
        self.total_loss += loss.item()
        self.count += 1

    def getBCELoss(self):
        return self.total_loss / self.count if self.count > 0 else 0