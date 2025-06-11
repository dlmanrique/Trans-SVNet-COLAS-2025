# This file process the LFB files and save the features for SAHC 

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import json
import argparse

import pickle




parser = argparse.ArgumentParser(description='Features formating')
parser.add_argument('--dataset', default='Cholec80', type=str)
args = parser.parse_args()


# Abrir archivo .pkl
with open(f"LFB/{args.dataset}/g_LFB50_train.pkl", "rb") as f:
    data_train = pickle.load(f)

# Abrir archivo .pkl
with open(f"LFB/{args.dataset}/g_LFB50_val.pkl", "rb") as f:
    data_valid = pickle.load(f)


with open(f"Video_ids/{args.dataset}/video_numbers.json", "r") as f:
    data_video_ids = json.load(f)


os.makedirs(f'Resnet_features_SAHC/{args.dataset}', exist_ok=True)

for vid in tqdm(list(dict.fromkeys(data_video_ids['Train']))):
    
    indices = [i for i, x in enumerate(data_video_ids['Train']) if x == vid]

    video_array = data_train[indices,:]
    np.save(f"Resnet_features_SAHC/{args.dataset}/{vid}_features.npy", video_array)



for vid in tqdm(list(dict.fromkeys(data_video_ids['Valid']))):
    
    indices = [i for i, x in enumerate(data_video_ids['Valid']) if x == vid]

    video_array = data_valid[indices,:]
    np.save(f"Resnet_features_SAHC/{args.dataset}/{vid}_features.npy", video_array)

