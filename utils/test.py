#!/usr/bin/env python
# coding=utf-8

'Test utils'

__author__ = 'Zhi Zhang, Mingjie Zheng, Sheng hua Zhong, Yan Liu'

import time
from os.path import isfile

import torch

from .dataset import SteganographyDataset


def test(model,
         criterion,
         root_folder='./data',
         folder_name='example',
         model_name='MLP',
         batch_size=100,
         parameters_name='',
         parameters_root_folder='./parameters',
         dataset='test',
         device='cuda',
         multi_target=1):
    # Dataset
    dataset = SteganographyDataset(root_folder=root_folder,
                                   dataset=dataset,
                                   folder_name=folder_name)
    dataset_size = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=4)

    # File name
    name = '{model_name}_{folder_name}'.format(model_name=model_name.lower(),
                                               folder_name=folder_name)
    if parameters_name:
        name = parameters_name

    # Read from model folder
    parameters_path = '{parameters_root_folder}/{name}.pt'.format(
        parameters_root_folder=parameters_root_folder, name=name)
    if isfile(parameters_path):
        model.load_state_dict(torch.load(parameters_path))

    since = time.time()

    # Test
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()

        # Forward
        # Track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.topk(outputs[:, 1], multi_target, 0)
            loss = criterion(outputs, labels)

        # Statistics
        # pdb.set_trace()
        running_loss += loss.item() * inputs.size(0)
        _, correct = torch.topk(labels[:, 1], multi_target, 0)
        # all right is right
        running_corrects += preds.equal(correct)

    loss = running_loss / dataset_size
    acc = running_corrects * batch_size / dataset_size

    print(
        "Total:",
        # "test_loss=", "{:.5f}".format(loss),
        "test_acc=",
        "{:.5f}".format(acc))

    time_elapsed = time.time() - since
    print("Summary:", "model is", "{model_name}".format(model_name=model_name),
          "dataset is", "{folder_name}".format(folder_name=folder_name))
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                       time_elapsed % 60))

    return model
