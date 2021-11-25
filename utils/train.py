#!/usr/bin/env python
# coding=utf-8

'Train utils'

__author__ = 'Zhi Zhang, Mingjie Zheng, Sheng hua Zhong, Yan Liu'

import copy
import time
from os.path import isfile

import torch
from tensorboardX import SummaryWriter

from .dataset import SteganographyDataset


def train(model,
          criterion,
          optimizer,
          scheduler,
          num_epochs=100,
          root_folder='./data',
          folder_name='example',
          model_name='MLP',
          parameters_name='',
          parameters_root_folder='./parameters',
          batch_size=100,
          summary=True,
          device='cuda'):
    # Dataset
    datasets = {
        x: SteganographyDataset(root_folder=root_folder,
                                dataset=x,
                                folder_name=folder_name)
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    dataloaders = {
        x: torch.utils.data.DataLoader(datasets[x],
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=4)
        for x in ['train', 'val']
    }

    # File name
    name = '{model_name}_{folder_name}'.format(model_name=model_name.lower(),
                                               folder_name=folder_name)
    if parameters_name:
        name = parameters_name

    # Summary
    if summary:
        writer = SummaryWriter(log_dir='./logs/{name}'.format(name=name))

    # Read from model folder
    parameters_path = '{parameters_root_folder}/{name}.pt'.format(
        parameters_root_folder=parameters_root_folder, name=name)
    if isfile(parameters_path):
        model.load_state_dict(torch.load(parameters_path))

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        loss_md = {x: 0 for x in ['train', 'val']}
        acc_md = {x: 0 for x in ['train', 'val']}

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # pdb.set_trace()
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, correct = torch.max(labels, 1)
                running_corrects += torch.sum(preds == correct)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # Mark down
            loss_md[phase] = epoch_loss
            acc_md[phase] = epoch_acc

        scheduler.step(loss_md['val'])

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=",
              "{:.5f}".format(loss_md['train']), "train_acc=",
              "{:.5f}".format(acc_md['train']), "val_loss=",
              "{:.5f}".format(loss_md['val']), "val_acc=",
              "{:.5f}".format(acc_md['val']))

        if summary:
            writer.add_scalar('train/loss', loss_md['train'], epoch)
            writer.add_scalar('train/acc', acc_md['train'], epoch)
            writer.add_scalar('val/loss', loss_md['val'], epoch)
            writer.add_scalar('val/acc', acc_md['val'], epoch)
            for name, param in model.named_parameters():
                writer.add_histogram('param_{name}'.format(name=name),
                                     param.data.clone().double().cpu().numpy(),
                                     epoch)
                writer.add_histogram('grads_{name}'.format(name=name),
                                     param.grad.clone().double().cpu().numpy(),
                                     epoch)

    time_elapsed = time.time() - since
    print("Summary:", "model is", "{model_name}".format(model_name=model_name),
          "dataset is", "{folder_name}".format(folder_name=folder_name))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                        time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save to model folder
    torch.save(best_model_wts, parameters_path)

    # Summary writer
    if summary:
        writer.close()
    return model
