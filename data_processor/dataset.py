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
            
            """인덱스 저장"""
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


# #%%
# # loading every get item
# class SingleShockDataset(Dataset):
#     def __init__(self, dataset_folder: Path,
#                  window_size: int=200, stride_size: int=1,
#                  start_percentage: float=0, end_percentage: float=1,
#                  sfreq:int=200, 
#                  pretrain_mode=True, dataset_split=None, rank=0):
#         self.__dataset_folder = dataset_folder
#         self.__dataset_name = str(self.__dataset_folder).split('/')[-1]
        
#         self.__header_path = dataset_folder / "channels.csv"
#         self.__label_path = dataset_folder / "labels.csv"
#         self.__npy_folder = dataset_folder / "npy_files"         
        
#         self.__window_size = window_size
#         self.__stride_size = stride_size
#         self.__start_percentage = start_percentage   # Index of percentage of the first sample of the dataset in the data file (inclusive)
#         self.__end_percentage = end_percentage       # Index of percentage of end of dataset sample in data file (not included)
#         self.__sfreq = sfreq
#         self.pretrain_mode = pretrain_mode
#         self.dataset_split = dataset_split
        
#         h5_save_path = f"{str(self.__dataset_folder)}/{self.__dataset_name}.h5"
#         if not pretrain_mode and dataset_split is not None:
#             # finetuning mode & already splitted
#             h5_save_path = f"{str(self.__dataset_folder)}/{self.__dataset_name}_{dataset_split}.h5"
#         multiple_label_files = list(self.__dataset_folder.glob("labels_*.csv"))
        

#         """ Load Preprocessed h5 Dataset OR cache/npz """
#         npz_save_path = h5_save_path.replace(".h5", ".npz")
#         pkl_save_path = h5_save_path.replace(".h5", ".pkl")
#         CACHE_PATH = npz_save_path.replace(".npz", ".pt")
        
#         if ('2018_PhysioNet_Challenge' in self.__dataset_name) or ('Siena' in self.__dataset_name) or ('TUEG' in self.__dataset_name):
#             base_dir = '/'.join(h5_save_path.split('/')[:-1])
#             edited_dir = '_'.join(base_dir.split('_')[:-1])
#             h5_save_path = os.path.join(edited_dir, h5_save_path.split('/')[-1])
            
#         # 변경: h5 파일이면 init 시에는 메타데이터만 읽고 닫음.
#         if os.path.exists(h5_save_path):
#             start_h5 = time.time()
#             self.use_h5_save_path = True
#             self._h5_save_path = h5_save_path
#             # read metadata and close
#             with h5py.File(h5_save_path, 'r') as f:
#                 if 'eeg' not in f:
#                     raise KeyError(f"'eeg' group not found in {h5_save_path}")
#                 keys = list(f['eeg'].keys())
#                 self._h5_keys = keys
#                 self._n_files = len(keys)
#                 self._file_lengths = []
#                 self._n_channels = None
#                 for key in keys:
#                     shape = f['eeg'][key].shape
#                     if self._n_channels is None and len(shape) >= 1:
#                         self._n_channels = shape[0]
#                     self._file_lengths.append(shape[1] if len(shape) > 1 else shape[0])
#                 # attrs (channels, labels) 가능하면 읽어 저장
#                 try:
#                     raw_channels = f.attrs.get("channels", None)
#                     if raw_channels is None:
#                         self.__ch_names = []
#                     elif isinstance(raw_channels, (list, np.ndarray)):
#                         self.__ch_names = list(raw_channels)
#                     else:
#                         self.__ch_names = json.loads(raw_channels)
#                 except Exception:
#                     self.__ch_names = []
#                 try:
#                     raw_labels = f.attrs.get("labels", None)
#                     if raw_labels is None:
#                         self.__labels = [] if self.pretrain_mode else []
#                     elif isinstance(raw_labels, (list, np.ndarray)):
#                         self.__labels = list(raw_labels)
#                     else:
#                         self.__labels = json.loads(raw_labels)
#                 except Exception:
#                     self.__labels = [] if self.pretrain_mode else []
#             end_h5 = time.time()
#             load_time = end_h5 - start_h5
#             if rank==0:
#                 print(f'Lazy metadata read from h5 file. Dataset meta loaded: {h5_save_path}')
#                 print(f"{self.__dataset_name} Dataset Initialized (meta only): {load_time:.3f}s taken")
#             # do not keep open file handle here
#             self.__files_eeg = None
#         else:
#             self.use_h5_save_path = False
#             # 기존 캐시/npz 로딩 (메모리 로드)
#             if os.path.exists(CACHE_PATH):
#                 start_cache = time.time()
#                 data = torch.load(CACHE_PATH, weights_only=True, map_location='cpu')
#                 self.__files_eeg = data["eeg"]
#                 self.__labels = data["labels"] if not self.pretrain_mode else []
#                 self.__ch_names = data["channels"]
#                 end_cache = time.time()
#                 load_time = end_cache - start_cache

#                 if rank==0:
#                     print(f"Dataset Loaded from cache: {CACHE_PATH}")
#                     print(f"{self.__dataset_name} Dataset Initialized: {load_time//60}m {load_time - load_time//60:.1f}s taken")
#                     print(self.__ch_names)
#                     print(len(self.__files_eeg))
#                     print(f'Sample shape example: {self.__files_eeg[0].shape}')
            
#             else:
#                 if ('2018_PhysioNet_Challenge' in self.__dataset_name) or ('Siena' in self.__dataset_name) or ('TUEG' in self.__dataset_name):
#                     start_npz = time.time()
#                     num_split_npz_files = {'2018_PhysioNet_Challenge':5,
#                                         'TUEG':24,
#                                         'Siena':11,
#                                         }[self.__dataset_name]
#                     self.__files_eeg = []
#                     for i in range(num_split_npz_files):
#                         data_i = np.load(npz_save_path.split('.')[0] + f"_{i}.npz", allow_pickle=True)
#                         eeg_data_i = list(data_i["eeg"])   # list of 2-d np.ndarray
#                         self.__files_eeg.extend(eeg_data_i)
#                     self.__labels = []
#                     self.__ch_names = list(data_i['channels'])
#                     end_npz = time.time()
#                 else:
#                     start_npz = time.time()
#                     data = np.load(npz_save_path, allow_pickle=True)
#                     self.__files_eeg = list(data["eeg"])
#                     self.__labels = [] if self.pretrain_mode else list(data["labels"])
#                     self.__ch_names = list(data["channels"])
#                     end_npz = time.time()
                
#                 load_time = end_npz - start_npz      
#                 if rank == 0:
#                     print(f"[NPZ] Dataset Loaded: {npz_save_path}")
#                     print(f"{self.__dataset_name} Dataset Initialized: {load_time//60}m {load_time - load_time//60:.1f}s taken")
#                     print(self.__ch_names)
#                     print(len(self.__files_eeg))
#                     print(f'Sample shape example: {self.__files_eeg[0].shape}')

#                     torch.save({
#                         'eeg':   self.__files_eeg,
#                         'labels':self.__labels,
#                         'channels': self.__ch_names
#                     }, CACHE_PATH, pickle_protocol=4)   ## ADDED (JW)

#         if not self.pretrain_mode:
#             if len(self.__labels) > 0:
#                 label_min_value = np.array(self.__labels).min()
#                 if label_min_value > 0:
#                     self.__labels = list(np.array(self.__labels) - label_min_value)

#         # initialize index structures
#         self.__length = None
#         self.__feature_size = None
#         self.__global_idxes = []
#         self.__local_idxes = []
#         self.__total_timestamps = 0
#         self.__init_dataset()
#         if rank == 0:
#             print(f"{self.__dataset_name} Dataset Initialized: {self.__length} training samples loaded. {self.__total_timestamps} timestamps ({self.__total_timestamps/(self.__sfreq * 3600) :.3f} hours).")

        
#     def __init_dataset(self):
#         # trials list: if using h5 metadata, range over keys; else over in-memory list
#         if self.use_h5_save_path:
#             self.__trials = list(range(len(self._h5_keys)))
#         else:
#             self.__trials = list(range(len(self.__files_eeg)))
#         global_idx = 0
#         for index in self.__trials:
#             # get file length without opening per-sample (use metadata or in-memory)
#             if self.use_h5_save_path:
#                 file_len = int(self._file_lengths[index])
#             else:
#                 eeg_data = self.__files_eeg[index]
#                 file_len = eeg_data.shape[1]
#             self.__global_idxes.append(global_idx) # the start index of the subject's sample in the dataset
#             self.__total_timestamps += file_len
#             # total number of samples for this file
#             total_sample_num = (file_len - self.__window_size) // self.__stride_size + 1
#             # apply start/end percentage
#             start_idx = int(total_sample_num * self.__start_percentage) * self.__stride_size 
#             end_idx = int(total_sample_num * self.__end_percentage - 1) * self.__stride_size
#             self.__local_idxes.append(start_idx)
#             added = max(0, (end_idx - start_idx) // self.__stride_size + 1)
#             global_idx += added
#         self.__length = global_idx
#         # feature size: channels x window
#         if self.use_h5_save_path:
#             n_ch = int(self._n_channels) if self._n_channels is not None else None
#             self.__feature_size = [n_ch, self.__window_size]
#         else:
#             self.__feature_size = [self.__files_eeg[0].shape[0], self.__window_size]

#     @property
#     def feature_size(self):
#         return self.__feature_size

#     def __len__(self):
#         return self.__length

#     def __getitem__(self, idx: int):
#         if idx < 0:
#             idx = len(self) + idx
#         if idx < 0 or idx >= len(self):
#             raise IndexError("index out of range")

#         file_idx = bisect.bisect(self.__global_idxes, idx) - 1
#         if file_idx < 0:
#             file_idx = 0
#         item_start_idx = (idx - self.__global_idxes[file_idx]) * self.__stride_size + self.__local_idxes[file_idx]
        
#         if self.use_h5_save_path:
#             # open and close per access (requested change)
#             with h5py.File(self._h5_save_path, 'r') as f:
#                 key = self._h5_keys[file_idx]
#                 arr = f['eeg'][key][:, item_start_idx:item_start_idx+self.__window_size]
#                 eeg = torch.tensor(np.asarray(arr)).float()
#         else:
#             eeg = torch.tensor(self.__files_eeg[file_idx][:, item_start_idx:item_start_idx+self.__window_size]).float()

#         if self.pretrain_mode:
#             return eeg
#         else:
#             label = self.__labels[file_idx] if len(self.__labels) > 0 else None
#             return eeg, label

#     def get_ch_names(self):
#         if self.use_h5_save_path:
#             return list(self.__ch_names)
#         else:
#             # previous behavior for in-memory dataset
#             try:
#                 return list(self.__files_eeg.attrs["channels"]) if isinstance(self.__files_eeg.attrs["channels"], (list, np.ndarray)) else json.loads(self.__files_eeg.attrs["channels"])
#             except Exception:
#                 return list(self.__ch_names)

#     def get_labels(self):
#         if self.use_h5_save_path:
#             return list(self.__labels) if len(self.__labels) > 0 else []
#         else:
#             try:
#                 return list(self.__files_eeg.attrs["labels"]) if isinstance(self.__files_eeg.attrs["labels"], (list, np.ndarray)) else json.loads(self.__files_eeg.attrs["labels"])
#             except Exception:
#                 return list(self.__labels) if len(self.__labels) > 0 else []

#     def get_total_timestamps(self):
#         return self.__total_timestamps



# %%
list_path = List[Path]
class ShockDataset(Dataset):
    """integrate multiple hdf5 files"""
    def __init__(self, dataset_folders: list_path, 
                 window_size: int=200, stride_size: int=1, 
                 start_percentage: float=0, end_percentage: float=1,
                 sfreq: int=200, pretrain_mode=True, dataset_split=None, rank=0):
        '''
        Arguments will be passed to SingleShockDataset. Refer to SingleShockDataset.
        '''
        self.__folder_paths = dataset_folders
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage
        
        self.__sfreq = sfreq
        self.pretrain_mode = pretrain_mode
        self.dataset_split = dataset_split
        self.rank = rank
        self.__datasets = []
        self.__length = None
        self.__feature_size = None

        self.__dataset_idxes = []
        
        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__datasets = [SingleShockDataset(dataset_folder=folder_path, 
                                              window_size=self.__window_size, 
                                              stride_size=self.__stride_size, 
                                              start_percentage=self.__start_percentage, 
                                              end_percentage=self.__end_percentage,
                                              sfreq=self.__sfreq,
                                              pretrain_mode=self.pretrain_mode,
                                              dataset_split=self.dataset_split,
                                              rank=self.rank,)
                           for folder_path in self.__folder_paths]
        
        # calculate the number of samples for each subdataset to form the integral indexes
        dataset_idx = 0
        for dataset in self.__datasets:
            self.__dataset_idxes.append(dataset_idx)
            dataset_idx += len(dataset)
        self.__length = dataset_idx

        self.__feature_size = self.__datasets[0].feature_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        dataset_idx = bisect.bisect(self.__dataset_idxes, idx) - 1
        item_idx = (idx - self.__dataset_idxes[dataset_idx])
        return self.__datasets[dataset_idx][item_idx]
    
    def get_ch_names(self):
        ch_names = self.__datasets[0].get_ch_names()
        input_chans = get_input_chans(ch_names)
        return input_chans

    def get_labels(self):
        return self.__datasets[0].get_labels()

    def get_total_timestamps(self):
        return sum(dataset.get_total_timestamps() for dataset in self.__datasets)
# %%
