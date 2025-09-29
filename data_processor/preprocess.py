#%%
import numpy as np
import pandas as pd
import bisect
import mne
from pathlib import Path
from typing import List
from torch.utils.data import Dataset
from dataset_utils import standard_1020
import h5py 
import os
import json
from collections import defaultdict
import re

import logging
import joblib

class Preprocess(Dataset):
    def __init__(self, dataset_folder: Path,
                 sfreq:int=200, pretrain_mode=True, 
                 dataset_type=1, dataset_split=None,
                 # dataset_type: (1=general, 2=splitted, 3=multiple_label)
                 num_chans=32
                 ):
        self.__dataset_folder = dataset_folder
        self.__dataset_name = str(self.__dataset_folder).split('/')[-1]
        
        self.__header_path = dataset_folder / "channels.csv"
        self.__label_path = dataset_folder / "labels.csv"
        self.__npy_folder = dataset_folder / "npy_files"         
        
        self.__sfreq = sfreq
        self.pretrain_mode = pretrain_mode
        self.dataset_split = dataset_split
        self.num_chans = num_chans
        
        h5_save_path = f"{str(self.__dataset_folder)}/{self.__dataset_name}.h5"
        if dataset_type == 1:
            self.__files_npy_list = list(self.__npy_folder.glob("*.npy"))   
            self.__labels = self.__load_labels() if not pretrain_mode else []
        elif dataset_type == 2:
            split_folders = {
                "train": "train_npy_files",
                "validation": "validation_npy_files",
                "test": "test_npy_files"
            }
            if pretrain_mode:
                self.__files_npy_list = []
                self.__labels = []
                for split, folder_name in split_folders.items():
                    split_folder_path = self.__dataset_folder / folder_name
                    label_path = self.__dataset_folder / f"{split}_labels.csv"
                    if split_folder_path.exists():
                        self.__files_npy_list.extend(list(split_folder_path.glob("*.npy")))
            else:
                h5_save_path = f"{str(self.__dataset_folder)}/{self.__dataset_name}_{dataset_split}.h5"
                split_folder_path = self.__dataset_folder / split_folders[dataset_split]
                self.__label_path = self.__dataset_folder / f"{dataset_split}_labels.csv"
                self.__files_npy_list = list(split_folder_path.glob("*.npy"))
                self.__labels = self.__load_labels() if not pretrain_mode else None
        elif dataset_type == 3:
            self.__files_npy_list = list(self.__npy_folder.glob("*.npy"))  
            multiple_label_files = list(self.__dataset_folder.glob("labels_*.csv"))
            assert len(multiple_label_files) > 0
            if not pretrain_mode:
                self.__labels = {}   
                for label_file in multiple_label_files:
                    label_name = label_file.stem.split("_")[-1]
                    self.__label_path = label_file
                    self.__labels[label_name] = self.__load_labels()
                self.__labels = json.dumps(self.__labels)
            else:
                self.__labels = []
        
        if Path(h5_save_path).exists():
            """ Load Preprocessed h5 Dataset """
            load_h5_dataset(h5_save_path, pretrain_mode, dataset_type)
        
        else:
            """ Preprocess Dataset"""
            self.__files_eeg = []
            self.__ch_names = None
            self.__init_preprocess_dataset()
            
            """Save as h5 format"""
            if not self.pretrain_mode:
                if dataset_type == 3:
                    label_temp = json.loads(self.__labels)
                    print("label_temp:", label_temp)
                    print(label_temp[list(label_temp.keys())[0]])
                    assert len(self.__files_eeg) == len(label_temp[list(label_temp.keys())[0]])
                else:
                    print("len(self.__files_eeg): ", len(self.__files_eeg))
                    print("len(self.__labels): ", len(self.__labels))
                    assert len(self.__files_eeg) == len(self.__labels)
            save_h5_dataset(h5_save_path, self.__files_eeg, self.__ch_names, self.__labels)
        
    
    def __init_preprocess_dataset(self):
        if not self.__files_npy_list:
            print(f"⚠ Warning: No .npy files found in {self.__npy_folder}")
            return
        
        raw_datasets = ["SEED"]
        if self.pretrain_mode and self.__dataset_name not in raw_datasets:
            # pretrain mode
            # concat same session and subject trials 
            concat_files_npy_list = concat_same_session_or_subject_trials(self.__files_npy_list, self.num_chans)
            print("concat_files_npy_list: ", len(concat_files_npy_list))
            for eeg_data in concat_files_npy_list:
                if eeg_data.shape[0] !
                eeg_data = self.preprocess_data(eeg_data)  
                self.__files_eeg.append(eeg_data)
            
        elif not self.pretrain_mode:
            # finetuning mode
            for npy_file in self.__files_npy_list:
                eeg_data = np.load(npy_file)
                if eeg_data.shape[0] != self.num_chans:
                    eeg_data = eeg_data.T 

                eeg_data = self.preprocess_data(eeg_data)  
                self.__files_eeg.append(eeg_data)
            
            
    def __load_header(self):
        if self.__header_path.exists():
            df = pd.read_csv(self.__header_path)
            header = [col.upper() for col in df.columns.tolist()]
            return header

    def __load_labels(self):
        if self.__label_path.exists():
            labels = pd.read_csv(self.__label_path, header=None)[1].tolist()
        return labels


    def preprocess_data(self, eeg_data, bandpass_low=0.3, bandpass_high=75.0, notch_filter_freq=50.0):

        sfreq = self.__sfreq  
        ch_names = self.__load_header()
        filtered_channels = [ch for ch in ch_names if ch in standard_1020]
        eeg_data = eeg_data[[ch_names.index(ch) for ch in filtered_channels]]
        self.__ch_names = filtered_channels

        eeg_data = np.nan_to_num(eeg_data)
        
        # - bandpass filter
        # - notch filter
        info = mne.create_info(ch_names=self.__ch_names, sfreq=sfreq, ch_types=['eeg'] * len(self.__ch_names), verbose=0)
        raw = mne.io.RawArray(eeg_data, info, verbose=0)
        raw.filter(l_freq=bandpass_low, h_freq=bandpass_high, verbose=0)
        raw.notch_filter(notch_filter_freq, verbose=0)

        raw.resample(sfreq, verbose=0)

        eeg_data = raw.get_data(units='uV', verbose=0)
        mean = np.mean(eeg_data, axis=-1, keepdims=True)
        std = np.std(eeg_data, axis=-1, keepdims=True)
        eeg_data = (eeg_data - mean) / (std+1e-9)
        return eeg_data




def load_h5_dataset(h5_path: str, pretrain_mode=True, data_type=1):
    multiple_label = True if data_type==3 else False
    with h5py.File(h5_path, 'r', swmr=True) as f:
        ch_names = list(f.attrs["channels"])
        try:
            labels = json.loads(f.attrs["labels"]) if not pretrain_mode and multiple_label else list(f.attrs["labels"])
        except:
            labels = [f["labels"][()]]
        files_eeg = [f["eeg"][str(idx)][()] for idx in range(len(f["eeg"].keys()))]
    if not pretrain_mode:
        if multiple_label:
            assert len(files_eeg) == len(labels[list(labels.keys())[0]])
        else:
            assert len(files_eeg) == len(labels)
    print("====================================")
    print(f"Dataset Loaded: {h5_path}")
    print("# of ch_names:", len(ch_names))
    print("# of labels:", len(labels))
    print("# of files_eeg:", len(files_eeg))
    print("0th file shape:", files_eeg[0].shape)
    print("1th file shape:", files_eeg[1].shape)
    print("====================================")

def save_h5_dataset(h5_path, files_eeg, ch_names, labels):
    with h5py.File(h5_path, 'w') as f:
        f.attrs["channels"] = ch_names
        try:
            f.attrs["labels"] = labels
        except:
            f.create_dataset("labels", data=labels)
        eeg_group = f.create_group("eeg")   
        for idx, eeg in enumerate(files_eeg):
            dataset = eeg_group.create_dataset(str(idx), data=eeg)  
    print(f"===Dataset Saved: {h5_path}===")



def extract_session_subject(filename):
    match = re.search(r"(sess\w*)?_(sub\d+)", filename)
    if match:
        session_id = match.group(1) if match.group(1) else "NO_SESSION"
        subject_id = match.group(2)
        return session_id, subject_id
    else:
        # 세션 정보 없이 sub만 있는 경우 처리 (sub###_trial###.npy)
        match = re.search(r"(sub\d+)", filename)
        if match:
            return "NO_SESSION", match.group(1)
    return None, None

def concat_same_session_or_subject_trials(npy_files, num_chans):

    session_subject_dict = defaultdict(list)

    for npy_file in npy_files:
        session_id, subject_id = extract_session_subject(npy_file.stem)
        if session_id and subject_id:
            key = f"{session_id}_{subject_id}"
            session_subject_dict[key].append(npy_file)

    concatenated_data = []
    for key, files in session_subject_dict.items():
        eeg_trials = [np.load(str(f)) for f in files]
        if eeg_trials[0].shape[0] == num_chans:
            eeg_trials = [eeg.T for eeg in eeg_trials]
        eeg_concat = np.concatenate(eeg_trials, axis=0) 
        concatenated_data.append(eeg_concat)
    return concatenated_data

