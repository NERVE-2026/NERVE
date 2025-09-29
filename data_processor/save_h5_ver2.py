#%%
from pathlib import Path
import os
from preprocess import Preprocess

#%%
# 서영
DATASET_GROUPS = {
    1: {  # general
        # 'Mumtaz2016': 19,
        # 'P300': 8,
        # 'SEED-IV': 62,
        # # -- fine-tuning --
        # 'HCI-Tagging_ERP': 32,
        # 'SEED-V': 62,
        # 'SEED-VIG': 17,   # NOTE: label 크기가 너무 커서 attr 속성으로는 저장이 안됨
    },
    2: {  # train/val/test splitted
        # 'BCIC2020-3': 64,
        # # -- fine-tuning --
        'BCI-NER Challenge': 56,
    },
    3: {  # multiple labels
        # 'DREAMER': 14,  
        # # -- fine-tuning --
        # 'DEAP': 32,
        # 'HCI-Tagging_emotion': 32,
    }
}

# 민성
# NOTE: 파일이름과 labels.csv의 파일이름 매칭 필요
# DATASET_GROUPS = {
#     1: {  # general
#         # 'Neonate': 19,
#         # 'Siena': 19,        # NOTE: 파일 중 데이터 채널 개수가 다른 게 있음 
#         # 'Fatigueset': 4,    # NOTE: Fatigueset/npy_files/sess02_sub02_trial01.npy' 중 '810.69599.010986328125' 에러
#         # 'Mental Arithmetic': 19,
#         # 'Stew': 14,           # NOTE: label None? (pretrain이라 라벨은 상관없긴 하지만)
#         # -- fine-tuning --
#         # 'NMT(Events)': 19,    # NOTE: 파일개수, 라벨개수 매칭 오류
#         # 'CHB-MIT': 23,      # NOTE: 채널개수 오류 (8개 drop됨)
#         # 'TUSL': # NOTE: 채널개수가 더 적음 
#     },
#     2: {  # train/val/test splitted
#         # -- fine-tuning --
#         'NMT(Scalp-EEG)': 19,   # NOTE: labels.csv -> train_labels.csv, test_labels.csv로 나누어야 함
#         'TUAB': 21,             # NOTE: labels.csv -> train_labels.csv, test_labels.csv로 나누어야 함
#         'TUEV': # NOTE: 채널개수가 더 적음 & labels.csv -> train_labels.csv, test_labels.csv로 나누어야 함
#         'TUSZ': 20              # NOTE: labels.csv -> train_labels.csv, test_labels.csv로 나누어야 함 & dev_npy_files은 무엇?
#     },
#     3: {  # multiple labels
#     }
# }

# # 영준
# # 채널매칭되는 것만 !!! 먼저 넣어서 저장
# DATASET_GROUPS = {
#     1: {  # general
        
#         # -- fine-tuning --

#     },
#     2: {  # train/val/test splitted
        
#         # -- fine-tuning --
#     },
#     3: {  # multiple labels
        
#     }
# }
FINETUNING_DATASETS = {
    # 서영
    'HCI-Tagging_ERP',
    'SEED-V',
    'SEED-VIG',
    'BCI-NER Challenge',
    'DEAP',
    'HCI-Tagging_emotion',
    # 민성
    'NMT(Scalp-EEG)',
    'CHB-MIT',
    'NMT(Events)',
}

def run_preprocessing(base_dir: str):
    for dataset_type, datasets in DATASET_GROUPS.items():
        for dataset_name, num_chans in datasets.items():
            dataset_folder = Path(base_dir) / dataset_name
            pretrain_mode = dataset_name not in FINETUNING_DATASETS
            dataset_split = 'train' if dataset_name in FINETUNING_DATASETS else None
            # dataset_split = 'test' if dataset_name in FINETUNING_DATASETS else None
            print(f"🚀 Processing {dataset_name} | Type: {dataset_type} | Pretrain: {pretrain_mode}")
            Preprocess(
                dataset_folder=dataset_folder,
                pretrain_mode=pretrain_mode,
                dataset_type=dataset_type,
                dataset_split=dataset_split,
                num_chans=num_chans
            )


if __name__ == '__main__':
    BASE_DIR = '/home/kjiwon0219/jiwon/Nas_folder_EEG/dataset/processed'
    run_preprocessing(BASE_DIR)




