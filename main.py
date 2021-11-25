#!/usr/bin/env python
# coding=utf-8

'Train and test model'

__author__ = 'Zhi Zhang, Mingjie Zheng, Sheng hua Zhong, Yan Liu'

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import models
from utils.test import test
from utils.train import train

# Params
parse = argparse.ArgumentParser()
parse.add_argument("--epochs",
                   help="Number of epochs to train.",
                   type=int,
                   default=10)
parse.add_argument("--model_name", help="Name of model", default='GCN')
parse.add_argument("--batch_size", help="Size of batch", type=int, default=100)
parse.add_argument("--folder_name",
                   help="Folder of experiment",
                   default='suniward_04_100')
parse.add_argument("--gpu", help="Order number of GPU", type=int, default=2)
parse.add_argument("--summary",
                   help="Summary of weights and grads",
                   action='store_true')
parse.add_argument("--mode", help="Mode of train or test", default='test')
parse.add_argument("--learning_rate",
                   help="Learning rate",
                   type=float,
                   default=1e-4)
parse.add_argument("--parameters_name",
                   help="Name of trained model",
                   default='')
parse.add_argument("--target_num",
                   help="Reset the folder or not",
                   type=int,
                   default=1)
parse.add_argument("--embidding_func",
                   help="Function of embidding",
                   default='Euclidean')
parse.add_argument("--step_size", help="Size of step", type=int, default=10)
args = parse.parse_args()

epochs = args.epochs
model_name = args.model_name
batch_size = args.batch_size
folder_name = args.folder_name
gpu = args.gpu
summary = args.summary
mode = args.mode
learning_rate = args.learning_rate
parameters_name = args.parameters_name
target_num = args.target_num
step_size = args.step_size

print("Setting:", f"epochs={epochs}", f"model_name={model_name}",
      f"batch_size={batch_size}", f"folder_name={folder_name}", f"gpu={gpu}",
      f"summary={summary}", f"mode={mode}", f"learning_rate={learning_rate}",
      f"parameters_name={parameters_name}", f"target_num={target_num}",
      f"step_size={step_size}")

# Use GPU
device = torch.device('cuda:{gpu}'.format(
    gpu=gpu) if torch.cuda.is_available() else 'cpu')

# Model and loss
if model_name == 'SAGCN':
    model_ft = models.SAGCN().to(device)
elif model_name == 'SAGCNMean':
    model_ft = models.SAGCNMean().to(device)
else:
    raise ValueError('Invalid argument for model: {model_name}'.format(
        model_name=model_name))

criterion = nn.SoftMarginLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(),
                         lr=learning_rate,
                         momentum=0.9,
                         weight_decay=5e-4)

# Decay LR by a factor of 0.1 every 20 epochs
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft,
                                                  patience=step_size,
                                                  factor=0.1)

# Train and evaluate / Test
if mode == 'train':
    print('Train:')
    model_ft = train(model_ft,
                     criterion,
                     optimizer_ft,
                     exp_lr_scheduler,
                     num_epochs=epochs,
                     folder_name=folder_name,
                     model_name=model_name,
                     batch_size=batch_size,
                     parameters_name=parameters_name,
                     summary=summary,
                     device=device)
elif mode == 'test':
    print('Test:')
    model_ft = test(model_ft,
                    criterion,
                    folder_name=folder_name,
                    model_name=model_name,
                    batch_size=batch_size,
                    parameters_name=parameters_name,
                    multi_target=target_num,
                    device=device)
else:
    raise ValueError('Invalid argument for mode: {mode}'.format(mode=mode))
