#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import shutil

from common.logger import Logger
from common.avgmeter import *
from common.sync_batchnorm.batchnorm import convert_model
from common.warmupLR import *
from tasks.semantic.modules.segmentator import *
from tasks.semantic.modules.mseEval import *

class Trainer():
  def __init__(self, ARCH, DATA, datadir, logdir, path=None):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.log = logdir
    self.path = path
    self.checkpoint_dir = os.path.join(os.path.dirname(self.log), "checkpoint")
    if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    self.train_mse_history = []  # 存储训练集MSE
    self.val_mse_history = []    # 存储验证集MSE    

    # 先初始化设备
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training in device: ", self.device)


    # put logger where it belongs
    self.tb_logger = Logger(self.log + "/tb")
    self.info = {"train_update": 0,
                 "train_loss": 0,
                 "train_mse": 0,
                 "valid_loss": 0,
                 "valid_mse": 0,
                 "backbone_lr": 0,
                 "decoder_lr": 0,
                 "head_lr": 0,
                 "post_lr": 0}

    # get the data
    parserPath = os.path.join(booger.TRAIN_PATH, "tasks", "semantic",  "dataset", self.DATA["name"], "parser.py")
    parserModule = imp.load_source("parserModule", parserPath)
    self.parser = parserModule.Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=None,
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=self.ARCH["train"]["batch_size"],
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=True)
    
    # 初始化模型
    with torch.no_grad():
        self.model = Segmentator(self.ARCH,
                                self.parser.get_n_classes(),
                                self.path).to(self.device)
        
        
    

    # # weights for loss (and bias)
    # epsilon_w = self.ARCH["train"]["epsilon_w"]
    # content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
    # for cl, freq in DATA["content"].items():
    #   x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
    #   content[x_cl] += freq
    # self.loss_w = 1 / (content + epsilon_w)   # get weights
    # for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
    #   if DATA["learning_ignore"][x_cl]:
    #     # don't weigh
    #     self.loss_w[x_cl] = 0
    # print("Loss weights from content: ", self.loss_w.data)

    # # concatenate the encoder and the head
    # with torch.no_grad():
    #   self.model = Segmentator(self.ARCH,
    #                            self.parser.get_n_classes(),
    #                            self.path)





    # GPU?
    self.gpu = False
    self.multi_gpu = False
    self.n_gpus = 0
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.n_gpus = 1
      self.model.cuda()
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      self.model = nn.DataParallel(self.model)   # spread in gpus
      self.model = convert_model(self.model).cuda()  # sync batchnorm
      self.model_single = self.model.module  # single model to get weight names
      self.multi_gpu = True
      self.n_gpus = torch.cuda.device_count()


    # 初始化 MSE 计算器
    self.mse_evaluator = mseEval(self.device) 



    #此处修改，分类的损失函数修改为回归的损失函数
    # loss
    if "loss" in self.ARCH["train"].keys() and self.ARCH["train"]["loss"] == "xentropy":
      #self.criterion = nn.NLLLoss(weight=self.loss_w).to(self.device)
      self.criterion = nn.MSELoss().to(self.device)
    else:
      raise Exception('Loss not defined in config file')
    # loss as dataparallel too (more images in batch)
    if self.n_gpus > 1:
      #self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpus
      self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpus

    # optimizer
    if self.ARCH["post"]["CRF"]["use"] and self.ARCH["post"]["CRF"]["train"]:
      self.lr_group_names = ["post_lr"]
      self.train_dicts = [{'params': self.model_single.CRF.parameters()}]
    else:
      self.lr_group_names = []
      self.train_dicts = []
    if self.ARCH["backbone"]["train"]:
      self.lr_group_names.append("backbone_lr")
      self.train_dicts.append(
          {'params': self.model_single.backbone.parameters()})
    if self.ARCH["decoder"]["train"]:
      self.lr_group_names.append("decoder_lr")
      self.train_dicts.append(
          {'params': self.model_single.decoder.parameters()})
    if self.ARCH["head"]["train"]:
      self.lr_group_names.append("head_lr")
      self.train_dicts.append({'params': self.model_single.head.parameters()})

    # Use SGD optimizer to train
    self.optimizer = optim.SGD(self.train_dicts,
                               lr=self.ARCH["train"]["lr"],
                               momentum=self.ARCH["train"]["momentum"],
                               weight_decay=self.ARCH["train"]["w_decay"])

    # Use warmup learning rate
    # post decay and step sizes come in epochs and we want it in steps
    steps_per_epoch = self.parser.get_train_size()
    up_steps = int(self.ARCH["train"]["wup_epochs"] * steps_per_epoch)
    final_decay = self.ARCH["train"]["lr_decay"] ** (1/steps_per_epoch)
    self.scheduler = warmupLR(optimizer=self.optimizer,
                              lr=self.ARCH["train"]["lr"],
                              warmup_steps=up_steps,
                              momentum=self.ARCH["train"]["momentum"],
                              decay=final_decay)
    

  @staticmethod
  def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    return color_range.reshape(256, 1, 3)

  @staticmethod
  def make_log_img(depth, mask):
    # input should be [depth, pred, gt]
    # make range image (normalized to 0,1 for saving)
    depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                           norm_type=cv2.NORM_MINMAX,
                           dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
    out_img = cv2.applyColorMap(
        depth, Trainer.get_mpl_colormap('viridis')) * mask[..., None]
    # # make label prediction
    # pred_color = color_fn((pred * mask).astype(np.int32))
    # out_img = np.concatenate([out_img, pred_color], axis=0)
    # # make label gt
    # gt_color = color_fn(gt)
    # out_img = np.concatenate([out_img, gt_color], axis=0)
    return (out_img).astype(np.uint8)

  @staticmethod
  def save_to_log(logdir, logger, info, epoch, w_summary=False, model=None, img_summary=False, imgs=[]):
    # save scalars
    for tag, value in info.items():
      logger.scalar_summary(tag, value, epoch)

    # save summaries of weights and biases
    if w_summary and model:
      for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
        if value.grad is not None:
          logger.histo_summary(
              tag + '/grad', value.grad.data.cpu().numpy(), epoch)

    if img_summary and len(imgs) > 0:
      directory = os.path.join(logdir, "predictions")
      if not os.path.isdir(directory):
        os.makedirs(directory)
      for i, img in enumerate(imgs):
        name = os.path.join(directory, str(i) + ".png")
        cv2.imwrite(name, img)

  
  def load_checkpoint(self, filename="checkpoint.pth.tar"):
      start_epoch = 0
      best_train_mse = float('inf')  
      best_val_mse = float('inf')
      checkpoint_path = os.path.join(self.checkpoint_dir, filename)
      
      if os.path.isfile(checkpoint_path):
          print(f"=> Loading checkpoint '{filename}'")
          checkpoint = torch.load(checkpoint_path)
          start_epoch = checkpoint['epoch']
          best_train_mse = checkpoint.get('best_train_mse', float('inf'))
          best_val_mse = checkpoint.get('best_val_mse', float('inf'))
          
          # 加载历史数据
          self.train_mse_history = checkpoint.get('train_mse_history', [])
          self.val_mse_history = checkpoint.get('val_mse_history', [])
          
          # 加载模型和优化器状态
          if 'model_state_dict' in checkpoint:
              self.model.load_state_dict(checkpoint['model_state_dict'])
          if 'optimizer_state_dict' in checkpoint:
              self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          
          print(f"=> Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
      else:
          print(f"=> No checkpoint found at '{filename}'")
      
      # 返回加载的状态，现在包括最佳训练MSE
      return start_epoch, best_train_mse, best_val_mse




  def save_checkpoint(self, state, filename="checkpoint.pth.tar", is_best=False):
      # 保存checkpoint到指定目录
      torch.save(state, os.path.join(self.checkpoint_dir, filename))
      
      # 如果这是到目前为止最好的模型，额外保存一份
      if is_best:
          shutil.copyfile(os.path.join(self.checkpoint_dir, filename),
                          os.path.join(self.checkpoint_dir, 'model_best.pth.tar'))


    



  def train(self):
    # accuracy and IoU stuff
    # best_train_iou = 0.0
    # best_val_iou = 0.0

    #此处修改，不需要原有的评估指标，改为 MSE（暂时待定）

    # self.ignore_class = []
    # for i, w in enumerate(self.loss_w):
    #   if w < 1e-10:
    #     self.ignore_class.append(i)
    #     print("Ignoring class ", i, " in IoU evaluation")
    # self.evaluator = iouEval(self.parser.get_n_classes(),
    #                          self.device, self.ignore_class)

    # 从checkpoint开始
    self.start_epoch, self.best_train_mse, self.best_val_mse = self.load_checkpoint()

    #初始化 MSE
    self.mse_evaluator

    # train for n epochs
    for epoch in range(self.start_epoch, self.ARCH["train"]["max_epochs"]):
      # get info for learn rate currently
      groups = self.optimizer.param_groups
      for name, g in zip(self.lr_group_names, groups):
        self.info[name] = g['lr']

      #重置 MSE 
      self.mse_evaluator.reset()

      # train for 1 epoch
      loss, update_mean = self.train_epoch(train_loader=self.parser.get_train_set(),
                          model=self.model,
                          criterion=self.criterion,
                          optimizer=self.optimizer,
                          epoch=epoch,
                          mse_evaluator=self.mse_evaluator,
                          scheduler=self.scheduler,
                          color_fn=self.parser.to_color,
                          report=self.ARCH["train"]["report_batch"],
                          show_scans=self.ARCH["train"]["show_scans"])


      # 获取并打印 MSE
      train_mse = self.mse_evaluator.getMSE()
      print(f"Epoch: {epoch}, Train MSE: {train_mse}")
      self.train_mse_history.append(train_mse) 
      
      # update info                                         
      self.info["train_loss"] = loss
      self.info["train_mse"] = train_mse


      if epoch % self.ARCH["train"]["report_epoch"] == 0:
        # evaluate on validation set
        print("*" * 80)
        loss, val_mse, rand_img = self.validate(val_loader=self.parser.get_valid_set(),
                                                 model=self.model,
                                                 criterion=self.criterion,
                                                 color_fn=self.parser.to_color,
                                                 save_scans=self.ARCH["train"]["save_scans"])

        # update info
        self.info["valid_loss"] = loss
        self.info["valid_mse"] = val_mse

        
        self.val_mse_history.append(val_mse)  


        print("*" * 80)

        # save to log
        Trainer.save_to_log(logdir=self.log,
                            logger=self.tb_logger,
                            info=self.info,
                            epoch=epoch,
                            w_summary=self.ARCH["train"]["save_summary"],
                            model=self.model_single,
                            img_summary=self.ARCH["train"]["save_scans"],
                            imgs=rand_img)
        
      current_state = {
          'epoch': epoch + 1,
          'model_state_dict': self.model.state_dict(),
          'optimizer_state_dict': self.optimizer.state_dict(),
          'best_train_mse': self.best_train_mse,
          'best_val_mse': self.best_val_mse,
          'train_mse_history': self.train_mse_history,
          'val_mse_history': self.val_mse_history,
      }

      # 检查当前模型是否为“最佳”模型，并保存
      is_best = False
      if train_mse < self.best_train_mse or val_mse < self.best_val_mse:
          is_best = True
          print("Best model so far, saving...")
          if train_mse < self.best_train_mse:
              self.best_train_mse = train_mse
          if val_mse < self.best_val_mse:
              self.best_val_mse = val_mse

      # 调用save_checkpoint来保存当前状态
      self.save_checkpoint(current_state, filename="checkpoint.pth.tar", is_best=is_best)

    epochs = range(1, len(self.train_mse_history) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, self.train_mse_history, label='Training MSE')
    plt.plot(epochs, self.val_mse_history, label='Validation MSE')
    plt.title('Training and Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()

    # 先保存图像，然后再显示图形
    plt.savefig(os.path.join(self.checkpoint_dir, 'MSE_curve.png'))
    plt.show()
    plt.close()

    print('Finished Training')

    return

  def train_epoch(self, train_loader, model, criterion, optimizer, epoch, mse_evaluator, scheduler, color_fn, report=10, show_scans=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #acc = AverageMeter()
    #iou = AverageMeter()
    update_ratio_meter = AverageMeter()

    # empty the cache to train now
    if self.gpu:
      torch.cuda.empty_cache()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in enumerate(train_loader):
        # measure data loading time
      data_time.update(time.time() - end)
      if not self.multi_gpu and self.gpu:
        in_vol = in_vol.cuda()
        proj_mask = proj_mask.cuda()
      if self.gpu:
        proj_labels = proj_labels.cuda(non_blocking=True).long()


      #此处修改为适应于回归任务的loss
      # compute output
      output = model(in_vol, proj_mask)
      proj_labels = proj_labels.unsqueeze(1).float()  # 增加一个维度以匹配通道数
      loss = self.criterion(output, proj_labels)  # 直接使用 MSE 损失

      # compute gradient and do SGD step
      optimizer.zero_grad()
      if self.n_gpus > 1:
        idx = torch.ones(self.n_gpus).cuda()
        loss.backward(idx)
      else:
        loss.backward()
      optimizer.step()

      # measure accuracy and record loss
      
      losses.update(loss.item(), in_vol.size(0))
      self.mse_evaluator.addBatch(output, proj_labels)
      # 在一个 epoch 结束后或在需要计算当前 MSE 的时候
      train_mse = self.mse_evaluator.getMSE()

      #loss = loss.mean()
      # with torch.no_grad():
      #   evaluator.reset()
      #   argmax = output.argmax(dim=1)
      #   evaluator.addBatch(argmax, proj_labels)
      #   accuracy = evaluator.getacc()
      #   jaccard, class_jaccard = evaluator.getIoU()
      # losses.update(loss.item(), in_vol.size(0))
      # acc.update(accuracy.item(), in_vol.size(0))
      # iou.update(jaccard.item(), in_vol.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      # get gradient updates and weights, so I can print the relationship of
      # their norms
      update_ratios = []
      for g in self.optimizer.param_groups:
        lr = g["lr"]
        for value in g["params"]:
          if value.grad is not None:
            w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
            update = np.linalg.norm(-max(lr, 1e-10) *
                                    value.grad.cpu().numpy().reshape((-1)))
            update_ratios.append(update / max(w, 1e-10))
      update_ratios = np.array(update_ratios)
      update_mean = update_ratios.mean()
      update_std = update_ratios.std()
      update_ratio_meter.update(update_mean)  # over the epoch

      if show_scans:
        # get the first scan in batch and project points
        mask_np = proj_mask[0].cpu().numpy()
        depth_np = in_vol[0][0].cpu().numpy()
        argmax = output.argmax(dim=1)
        # pred_np = argmax[0].cpu().numpy()
        # gt_np = proj_labels[0].cpu().numpy()
        out = Trainer.make_log_img(depth_np, mask_np)
        cv2.imshow("sample_training", out)
        cv2.waitKey(1)


      # 此处修改，移除了准确率和IoU代码，增加mse_meter
      if i % self.ARCH["train"]["report_batch"] == 0:
        print('Lr: {lr:.3e} | '
              'Update: {umean:.3e} mean,{ustd:.3e} std | '
              'Epoch: [{0}][{1}/{2}] | '
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
              'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
              'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
              'MSE {train_mse:.4f}'.format(
              epoch, i, len(train_loader), batch_time=batch_time,
              data_time=data_time, loss=losses, lr=lr,
              umean=update_mean, ustd=update_std,
              train_mse=train_mse))

      # step scheduler
      scheduler.step()

    return losses.avg, update_ratio_meter.avg

  def validate(self, val_loader, model, criterion, color_fn, save_scans):
    batch_time = AverageMeter()
    losses = AverageMeter()
    rand_imgs = []

    # switch to evaluate mode
    model.eval()
    self.mse_evaluator.reset()

    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()
      for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in enumerate(val_loader):
        if not self.multi_gpu and self.gpu:
          in_vol = in_vol.cuda()
          proj_mask = proj_mask.cuda()
        if self.gpu:
          proj_labels = proj_labels.cuda(non_blocking=True).long()


        #此处修改为回归函数的计算方式
        # compute output
        output = model(in_vol, proj_mask)
        proj_labels = proj_labels.unsqueeze(1).float()  # 增加一个维度以匹配通道数
        loss = self.criterion(output, proj_labels)  # 直接使用 MSE 损失

        # measure accuracy and record loss
        # argmax = output.argmax(dim=1)
        # evaluator.addBatch(argmax, proj_labels)
        losses.update(loss.item(), in_vol.size(0))
        self.mse_evaluator.addBatch(output, proj_labels)

        if save_scans:
          # get the first scan in batch and project points
          mask_np = proj_mask[0].cpu().numpy()
          depth_np = in_vol[0][0].cpu().numpy()
          argmax = output.argmax(dim=1)
          # pred_np = argmax[0].cpu().numpy()
          # gt_np = proj_labels[0].cpu().numpy()
          out = Trainer.make_log_img(depth_np, mask_np)

          rand_imgs.append(out)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

      val_mse = self.mse_evaluator.getMSE()

      print('Validation set:\n'
            'Time avg per batch {batch_time.avg:.3f}\n'
            'Loss avg {loss.avg:.4f}\n'
            'MSE {val_mse:.4f}'.format(batch_time=batch_time,
                                      loss=losses,
                                      val_mse=val_mse))

    # 返回损失平均值和 MSE
    return losses.avg, val_mse, rand_imgs