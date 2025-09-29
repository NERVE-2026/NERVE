#%%
import numpy as np
import pandas as pd
import bisect
import mne
from pathlib import Path
from typing import List
from torch.utils.data import Dataset
from data_processor.dataset_utils import standard_1020, get_input_chans

import h5py 
import os
import json
import time
import logging   
import joblib
import torch
from tqdm import tqdm
#%%
# # one time loading h5
class SingleShockDataset(Dataset):
    def __init__(self, dataset_folder: Path,
                 window_size: int=200, stride_size: int=1,
                 start_percentage: float=0, end_percentage: float=1,
                 sfreq:int=200, 
                 pretrain_mode=True, dataset_split=None, rank=0):
        self.__dataset_folder = dataset_folder
        self.__dataset_name = str(self.__dataset_folder).split('/')[-1]
        
        self.__header_path = dataset_folder / "channels.csv"
        self.__label_path = dataset_folder / "labels.csv"
        self.__npy_folder = dataset_folder / "npy_files"         
        
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage   # Index of percentage of the first sample of the dataset in the data file (inclusive)
        self.__end_percentage = end_percentage       # Index of percentage of end of dataset sample in data file (not included)
        self.__sfreq = sfreq
        self.pretrain_mode = pretrain_mode
        self.dataset_split = dataset_split
        self.rank = rank
        
        h5_save_path = f"{str(self.__dataset_folder)}/{self.__dataset_name}.h5"
        if not pretrain_mode and dataset_split is not None:
            # finetuning mode & already splitted
            h5_save_path = f"{str(self.__dataset_folder)}/{self.__dataset_name}_{dataset_split}.h5"
        multiple_label_files = list(self.__dataset_folder.glob("labels_*.csv"))
        

        """ Load Preprocessed h5 Dataset """
        npz_save_path = h5_save_path.replace(".h5", ".npz")
        pkl_save_path = h5_save_path.replace(".h5", ".pkl")
        CACHE_PATH = npz_save_path.replace(".npz", ".pt")
        
        if ('2018_PhysioNet_Challenge' in self.__dataset_name) or ('Siena' in self.__dataset_name) or ('TUEG' in self.__dataset_name):
            base_dir = '/'.join(h5_save_path.split('/')[:-1])
            edited_dir = '_'.join(base_dir.split('_')[:-1])
            h5_save_path = os.path.join(edited_dir, h5_save_path.split('/')[-1])
            
        if os.path.exists(h5_save_path):
            start_h5 = time.time()
            self.use_h5_save_path = True
            f = h5py.File(h5_save_path, 'r', swmr=True, locking=False)
            
            self.__ch_names = list(f.attrs["channels"]) if isinstance(f.attrs["channels"], (list, np.ndarray)) else json.loads(f.attrs["channels"])
            self.__files_eeg = f["eeg"]
            if 'labels' in f.attrs.keys():
                self.__labels = list(f.attrs["labels"]) if isinstance(f.attrs["labels"], (list, np.ndarray)) else json.loads(f.attrs["labels"])
            else:
                self.__labels = list(f["labels"][()])

            if rank==0:
                print(f'Lazy loading from h5 file. Dataset Loaded from h5: {h5_save_path}')
                print(f"{self.__dataset_name} Dataset Initialized")
                print(self.__ch_names)

        else:
            if rank==0:
                print(f'h5 file does not exist: {h5_save_path}')

        if not self.pretrain_mode:
            # assert len(self.__files_eeg) == len(self.__labels)
            if 'SEED-VIG' in self.__dataset_name:
                self.__labels = list(np.array(self.__labels))
            else:
                label_min_value = np.array(self.__labels).min()
                if label_min_value > 0:
                    self.__labels = list(np.array(self.__labels) - label_min_value)

        # initialize
        self.__length = None
        self.__feature_size = None
        self.__global_idxes = []
        self.__local_idxes = []
        self.__total_timestamps = 0
        self.__init_dataset()

        end_h5 = time.time()
        load_time = end_h5 - start_h5
        if rank == 0:
            print(f"{self.__dataset_name} Dataset Initialized (time: {load_time:.1f}s): {self.__length} training samples loaded. {self.__total_timestamps} timestamps ({self.__total_timestamps/(self.__sfreq * 3600) :.3f} hours).")

        
    def __init_dataset(self):
        self.__trials = list(range(len(self.__files_eeg))) if self.use_h5_save_path else list(range(len(self.__files_eeg)))

        global_idx = 0
        for index in tqdm(self.__trials, disable=(self.rank != 0)):
            eeg_data = self.__files_eeg[str(index)] if self.use_h5_save_path else self.__files_eeg[index]
            
            self.__global_idxes.append(global_idx) # the start index of the subject's sample in the dataset
            file_len = eeg_data.shape[1]
            self.__total_timestamps += file_len
            # total number of samples
            total_sample_num = (file_len-self.__window_size) // self.__stride_size + 1
            # cut out part of samples
            start_idx = int(total_sample_num * self.__start_percentage) * self.__stride_size 
            end_idx = int(total_sample_num * self.__end_percentage - 1) * self.__stride_size
            
            self.__local_idxes.append(start_idx)
            global_idx += (end_idx - start_idx) // self.__stride_size + 1
        
        self.__length = global_idx
        self.__feature_size = [eeg_data.shape[0], self.__window_size]

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        file_idx = bisect.bisect(self.__global_idxes, idx) - 1
        item_start_idx = (idx - self.__global_idxes[file_idx]) * self.__stride_size + self.__local_idxes[file_idx]
        
        eeg_file_idx = str(file_idx) if self.use_h5_save_path else file_idx
        eeg = torch.tensor(self.__files_eeg[eeg_file_idx][:, item_start_idx:item_start_idx+self.__window_size]).float()

        if self.pretrain_mode:
            return eeg
            
        else:
            label = self.__labels[file_idx]
            return eeg, label

    def get_ch_names(self):
        return self.__ch_names
    
    def get_labels(self):
        return self.__labels if self.__dataset_name!='SEED-VIG' else []

    def get_total_timestamps(self):
        return self.__total_timestamps