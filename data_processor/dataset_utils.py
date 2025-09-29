#%%
import os
from pathlib import Path
import pandas as pd
import numpy as np

# CHANNEL
standard_1020 = [
    'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', 
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', 'TP9', 'TP7', 
    'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', 
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', 
    'O1', 'OZ', 'O2', 'CB1', 'CB2', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', 
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', 
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', 'T1', 'T2', 
    'FP1-F7', 'FP2-F8', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'F7-T3', 'T3-T5', 
    'T5-O1', 'F8-T4', 'T4-T6', 'T6-O2', 'A1-T3', 'T3-C3', 'C3-CZ',  
    'T4-A2', 'I1', 'I2', 'IZ']
standard_1020 = [ch.upper() for ch in standard_1020]


def build_pretraining_dataset(base_dir: str, datasets: list, 
                              time_window: list, stride_size=200, 
                              start_percentage=0, end_percentage=1,
                              sfreq=200, rank=0):
    from data_processor.dataset import ShockDataset
    shock_dataset_list = []
    ch_index_list = []
    for dataset_list, window_size in zip(datasets, time_window):
        dataset = ShockDataset([Path(os.path.join(base_dir, folder_path)) for folder_path in dataset_list], 
                               window_size * 200, stride_size, 
                               start_percentage, end_percentage,
                               sfreq, rank=rank)
        shock_dataset_list.append(dataset)
        ch_index_list.append(dataset.get_ch_names())
    return shock_dataset_list, ch_index_list


def build_finetuning_dataset(base_dir: str, dataset: str, 
                             time_window: int, stride_size=200, 
                             start_percentage=0, end_percentage=1,
                             sfreq=200,
                             dataset_split=None, rank=0):
    from data_processor.dataset import ShockDataset
    # train,val,test 나눠진 경우에는
    dataset = ShockDataset([Path(os.path.join(base_dir, dataset))],
                        time_window * 200, stride_size,
                        start_percentage, end_percentage,
                        sfreq, 
                        pretrain_mode=False, 
                        dataset_split=dataset_split, rank=rank)
    ch_idx = dataset.get_ch_names()
    return dataset, ch_idx


def get_input_chans(ch_names):
    return [standard_1020.index(ch_name) for ch_name in ch_names if ch_name in standard_1020]
    input_chans = [0] # for cls token
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name) + 1)
    return input_chans

