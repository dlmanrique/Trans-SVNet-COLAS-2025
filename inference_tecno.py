#some codes adapted from https://github.com/YuemingJin/MTRCNet-CL
# and https://github.com/YuemingJin/TMRNet
# and https://github.com/tobiascz/TeCNO
import torch
from torch import optim
from torch import nn
import numpy as np
import mstcn
import pickle, time
import random
from sklearn import metrics
import copy
import wandb
import argparse
from tqdm import tqdm
import os
from evaluation_relaxed_metrics import relaxed_evaluation_M2CAI, relaxed_evaluation_Cholec80
import json


seed = 1
print("Random Seed: ", seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

parser = argparse.ArgumentParser(description="Experimento con W&B y MSTCN")
parser.add_argument("--dataset", type=str, default=None, help="Nombre del dataset")
parser.add_argument("--num_classes", type=int, default=7, help="Número de clases del dataset")
parser.add_argument("--batch", type=int, default=400)
parser.add_argument("--mstcn_causal_conv", type=bool, default=True, help="Usar convoluciones causales en MSTCN")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="Tasa de aprendizaje")
parser.add_argument("--max_epochs", type=int, default=25, help="Número máximo de épocas")
parser.add_argument("--mstcn_layers", type=int, default=8, help="Número de capas de MSTCN")
parser.add_argument("--mstcn_f_maps", type=int, default=32, help="Número de mapas de características en MSTCN")
parser.add_argument("--mstcn_f_dim", type=int, default=2048, help="Shape de los features del MViT")
parser.add_argument("--mstcn_stages", type=int, default=2, help="Número de etapas en MSTCN")
parser.add_argument("--inference", type=str, default='True')
parser.add_argument('--model_path', type=str, default='')
args = parser.parse_args()


if args.dataset == 'M2CAI':
    args.num_classes = 8
elif args.dataset == 'HeiCo':
    args.num_classes = 14
else:
    args.num_classes = 7

# Configure model path 
args.model_path = f'TeCNO50_models/{args.dataset}/TeCNO50_best_model.pth'


# Inicializar W&B con los argumentos como configuración
current_time = time.localtime()

wandb.init(
    project="TeCNO50 - COLAS 2025",
    entity='endovis_bcv', 
    name=time.strftime("%Y-%m-%d %H:%M:%S", current_time),
    config=vars(args) # Guarda todos los argumentos en wandb.config
)

def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    
    train_paths_80 = train_test_paths_labels[0]
    val_paths_80 = train_test_paths_labels[1]

    train_labels_80 = train_test_paths_labels[2]
    val_labels_80 = train_test_paths_labels[3]

    train_num_each_80 = train_test_paths_labels[4]
    val_num_each_80 = train_test_paths_labels[5]

    """train_num_each_80 = [train_test_paths_labels[4][0]]
    val_num_each_80 = [train_test_paths_labels[5][0]]

    train_paths_80 = train_test_paths_labels[0][:train_num_each_80[0]]
    val_paths_80 = train_test_paths_labels[1][:val_num_each_80[0]]
    train_labels_80 = train_test_paths_labels[2][:train_num_each_80[0]]
    val_labels_80 = train_test_paths_labels[3][:val_num_each_80[0]]"""


    print('train_paths_80  : {:6d}'.format(len(train_paths_80)))
    print('train_labels_80 : {:6d}'.format(len(train_labels_80)))
    print('valid_paths_80  : {:6d}'.format(len(val_paths_80)))
    print('valid_labels_80 : {:6d}'.format(len(val_labels_80)))

    val_labels_80 = np.asarray(val_labels_80, dtype=np.int64)


    val_start_vidx = []
    count = 0
    for i in range(len(val_num_each_80)):
        val_start_vidx.append(count)
        count += val_num_each_80[i]


    return val_labels_80, val_num_each_80, val_start_vidx, val_paths_80


def get_long_feature(start_index, lfb, LFB_length):
    long_feature = []
    long_feature_each = []
    # 上一个存在feature的index
    for k in range(LFB_length):
        LFB_index = (start_index + k)
        LFB_index = int(LFB_index)
        long_feature_each.append(lfb[LFB_index])
    long_feature.append(long_feature_each)
    return long_feature


test_labels_80, test_num_each_80, test_start_vidx, test_paths  = get_data(f'pkl_datasets_files/train_test_paths_labels_{args.dataset}.pkl')



with open(f"LFB/{args.dataset}/g_LFB50_test.pkl", 'rb') as f:
    g_LFB_test = pickle.load(f)


print("load completed")

print("g_LFB_test shape:", g_LFB_test.shape)



model = mstcn.MultiStageModel(args.mstcn_stages, args.mstcn_layers, args.mstcn_f_maps,
                               args.mstcn_f_dim, args.num_classes, args.mstcn_causal_conv)
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint)

model.cuda()


num_videos_dataset = {'Autolaparo': [10, 7], #-> values for train and test
                      'Cholec80': [40, 40],
                      'HeiChole': [16, 8],
                      'HeiCo': [21, 9],
                      'M2CAI': [27, 14]}

#Information used for saving the id of each prediction and perform relaxed evaluation,
#  its added to the video_id used in validation loop
gap_for_dataset = {'Cholec80': 62,
                   'M2CAI': 183}


test_we_use_start_idx_80 = [x for x in range(num_videos_dataset[args.dataset][1])]


print("Test videos:", g_LFB_test.shape)


# Sets the module in evaluation mode.
model.eval()

val_acc_each_video = []
val_all_preds_phase = []
val_all_labels_phase = []
all_preds_info = {}


with torch.no_grad():
    for i in tqdm(test_we_use_start_idx_80):

        labels_phase = []
        paths = []
        
        for j in range(test_start_vidx[i], test_start_vidx[i] + test_num_each_80[i]):
            labels_phase.append(test_labels_80[j])
            paths.append(test_paths[j])

        labels_phase = torch.LongTensor(labels_phase)

        if use_gpu:
            labels_phase = labels_phase.to(device)
        else:
            labels_phase = labels_phase

        long_feature = get_long_feature(start_index=test_start_vidx[i],
                                        lfb=g_LFB_test, LFB_length=test_num_each_80[i])

        long_feature = (torch.Tensor(long_feature)).to(device)
        video_fe = long_feature.transpose(2, 1)

        y_classes = model.forward(video_fe)

        p_classes = y_classes[-1].squeeze().transpose(1, 0)

        # Save the info in the dict and later save the dict as json file
        # Iterate over the file names (keys) and p_classes (values)
        for idx, img_path in enumerate(paths):
            all_preds_info[img_path] = p_classes[idx].cpu().numpy().tolist()


        #Sanity check
        stages = y_classes.shape[0]
        _, preds_phase = torch.max(y_classes[stages - 1].squeeze().transpose(1, 0).data, 1)

        for j in range(len(preds_phase)):
            val_all_preds_phase.append(int(preds_phase.data.cpu()[j]))

        for j in range(len(labels_phase)):
            val_all_labels_phase.append(int(labels_phase.data.cpu()[j]))


        val_acc_each_video.append(float(torch.sum(preds_phase == labels_phase.data))/test_num_each_80[i])

    val_acc_video = np.mean(val_acc_each_video)
    val_recall_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average='macro')
    val_precision_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average='macro')


breakpoint()
os.makedirs(f'TeCNO_preds/{args.dataset}')
with open(f'TeCNO_preds/{args.dataset}/preds_phases.json', "w", encoding="utf-8") as f:
    json.dump(all_preds_info, f, ensure_ascii=False, indent=4)

print(f'Accuracy: {val_acc_video}')
print(f'Precision: {val_precision_phase}')
print(f'Recall: {val_recall_phase}')










