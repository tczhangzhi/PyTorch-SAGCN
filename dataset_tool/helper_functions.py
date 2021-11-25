#!/usr/bin/env python
# coding=utf-8

'Define helper functions'

import numpy as np
from tqdm import tqdm

def vstack(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return np.vstack((a, b))

def hstack(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return np.hstack((a, b))

def safe_shuffle(data):
    data = data.copy()
    np.random.shuffle(data)
    return data

def shuffle_same_order(x, y):
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)
    return x, y

def batch_data(data):
    np.random.shuffle(data)
    return data.reshape(100, 200, 320)

def safe_batch_data(data):
    data = safe_shuffle(data)
    return data.reshape(100, 200, 320)

def mix_normal(guilty_data, normal_data, normal_prop):
    print('Mix normal in guilty:')
    for i in tqdm(range(len(guilty_data))):
        guilty_data[i][:normal_prop] = normal_data[i][:normal_prop]

        permutation = np.random.permutation(guilty_data[i].shape[0])
        guilty_data[i] = guilty_data[i][permutation, :]
    return guilty_data

def dataset(data, batch):
    result = None
    for _ in tqdm(range(batch)):
        result = vstack(result, batch_data(data))
    return result    