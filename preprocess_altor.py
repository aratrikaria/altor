 #!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/27 15:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : hhar.py
# @Description : http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition

import os
import numpy as np
import pandas as pd


def extract_sensor(data, time_index, time_tag, window_time):
    index = time_index
    while index < len(data) and abs(data.iloc[index]['Creation_Time'] - time_tag) < window_time:
        index += 1
    if index == time_index:
        return None, index
    else:
        data_slice = data.iloc[time_index:index]
        if data_slice['User'].unique().size > 1 or data_slice['gt'].unique().size > 1:
            return None, index
        else:
            data_sensor = data_slice[['x', 'y', 'z']].to_numpy()
            sensor = np.mean(data_sensor, axis=0)
            label = data_slice[['User', 'Model', 'gt']].iloc[0].values
            return np.concatenate([sensor, label]), index


accs = pd.read_csv(os.path.join(ALTOR_DATASET_PATH,'accs.csv'))
gyros = pd.read_csv(os.path.join(ALTOR_DATASET_PATH,'gyros.csv'))


##Preprocess only for Altor data
# 'Index', 'Arrival_Time', 'Creation_Time', 'x', 'y', 'z', 'User', 'Model', 'Device', 'gt'
def preprocess_altor(accs, gyros, window_time=500, seq_len=120, jump=0):
    #accs = pd.read_csv(path + '\Phones_accelerometer.csv')
    #gyros = pd.read_csv(path + '\Phones_gyroscope.csv') #, nrows=200000
    time_tag = min(accs.iloc[0, 2], gyros.iloc[0, 2])
    time_index = [0, 0] # acc, gyro
    window_num = 0
    data = []
    data_temp = []
    while time_index[0] < len(accs) and time_index[1] < len(gyros):
        acc, time_index_new_acc = extract_sensor(accs, time_index[0], time_tag, window_time)
        gyro, time_index_new_gyro = extract_sensor(gyros, time_index[1], time_tag, window_time)
        time_index = [time_index_new_acc, time_index_new_gyro]
        if acc is not None and gyro is not None and np.all(acc[-3:] == gyro[-3:]):
            time_tag += window_time
            window_num += 1
            #Each data point is ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z, USER, MODEL, GT
            data_temp.append(np.concatenate([acc[:-3], gyro[:-3], acc[-3:]]))
            if window_num == seq_len:
                data.append(np.array(data_temp))
                if jump == 0:
                    data_temp.clear()
                    window_num = 0
                else:
                    data_temp = data_temp[-jump:]
                    window_num -= jump
        else:
            if window_num > 0:
                data_temp.clear()
                window_num = 0
            if time_index[0] < len(accs) and time_index[1] < len(gyros):
                time_tag = min(accs.iloc[time_index[0], 2], gyros.iloc[time_index[1], 2])
            else:
                break
    return data

data = preprocess_altor(accs, gyros)


# label: ('User', 'Model', 'gt')
# ['nexus4' 's3' 's3mini']
# ['bike' 'sit' 'stairsdown' 'stairsup' 'stand' 'walk']

## For the HHAR Data, just laod the output of the pre processed data [separated data and label] from HHAR to stack
## data numpy and label numpy with altor
##paste your own local path after you download this numpys from below url and then load: HHAR_DATA_NPY, HHAR_LABEL_NPY
HHAR_DATASET_PATH = "https://github.com/dapowan/LIMU-BERT-Public/tree/master/dataset/hhar"
HHAR_DATA_NPY = os.path.join(HHAR_DATASET_PATH, "hhar_data_20_120.npy")
HHAR_LABEL_NPY = os.path.join(HHAR_DATASET_PATH, "hhar_label_20_120.npy")

hhar_data_np, hhar_label_np = np.load(HHAR_DATA_NPY), np.load(HHAR_LABEL_NPY)

#Converted the label to Alphabet since user was in numbers, while stacking, needed unique hence converted to alphabets
def transform_to_alphabet(label):
    labels_unique = np.unique(label)
    for i in range(labels_unique.size):
        label[label == labels_unique[i]] = chr(i+65)
    return label


def transform_to_index(label, hhar_label_size, print_label = False):
    labels_unique = np.np.unique(label)
    for i in range(labels_unique.size):
        label[label == labels_unique[i]] = i + hhar_label_size
        


def separate_data_label(altor_data_raw, hhar_data_label_raw):
    labels = altor_data_raw[:, :, -3:].astype(np.str)
    transform_to_index(transform_to_alphabet(labels[:, :, 0]), np.unique(hhar_data_label_raw[:, :, 0]).size)
    transform_to_index(labels[:, :, 1], np.unique(hhar_data_label_raw[:, :, 1]).size)
    transform_to_index(labels[:, :, 2], np.unique(hhar_data_label_raw[:, :, 2]).size)
    data = altor_data_raw[:, :, 6].astype(np.float)
    labels = labels.astype(np.float)
    return data, labels

def save_complete_data(path_save, version, altor_data):
    altor_data_raw = np.array(altor_data)
    altor_data, altor_label = separate_data_label(altor_data_raw, hhar_label_np)
    altor_data_np, altor_label_np = np.array(altor_data), np.array(altor_label)
    data_new, label_new = np.concatenate((hhar_data_np, altor_data_np)), np.concatenate((hhar_label_np, altor_label_np))
    
    np.save(os.path.join(path_save, 'hhar_altor_data_' + version + '.npy'), data_new)
    np.save(os.path.join(path_save, 'hhar_altor_label_' + version + '.npy'), label_new)
    return data_new, label_new

data_new, label_new = save_complete_data(OUTPUT_PATH, "20_120", data1)
# acc + gyro
#path_save = 'hhar'
#version = '20_120'
#data, label = preprocess_hhar(DATASET_PATH, path_save, version, window_time=50, seq_len=120)
