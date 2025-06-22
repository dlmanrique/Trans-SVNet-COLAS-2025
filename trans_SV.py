#some code adapted from https://github.com/YuemingJin/MTRCNet-CL
# and https://github.com/YuemingJin/TMRNet

import torch
from torch import optim
from torch import nn
import numpy as np
import pickle, time
import random
from sklearn import metrics
import copy
import mstcn
from transformer2_3_1 import Transformer2_3_1
from evaluation_relaxed_metrics import relaxed_evaluation_M2CAI, relaxed_evaluation_Cholec80

import wandb
import argparse
import os
from tqdm import tqdm


seed = 1
print("Random Seed: ", seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

parser = argparse.ArgumentParser(description="Training script for gene expression prediction.")
parser.add_argument("--dataset", type=str, default=None, help="Nombre del dataset")
parser.add_argument("--out_features", type=int, default=7, help="Number of output features.")
parser.add_argument("--num_workers", type=int, default=3, help="Number of workers for data loading.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
parser.add_argument("--mstcn_causal_conv", type=lambda x: x.lower() == 'true', default=True, help="Use causal convolution in MS-TCN.")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimization.")
parser.add_argument("--max_epochs", type=int, default=25, help="Maximum number of training epochs.")
parser.add_argument("--mstcn_layers", type=int, default=8, help="Number of layers in MS-TCN.")
parser.add_argument("--mstcn_f_maps", type=int, default=32, help="Feature maps in MS-TCN.")
parser.add_argument("--mstcn_f_dim", type=int, default=2048, help="Feature dimension in MS-TCN.")
parser.add_argument("--mstcn_stages", type=int, default=2, help="Number of stages in MS-TCN.")
parser.add_argument("--sequence_length", type=int, default=30, help="Length of the input sequence.")

args = parser.parse_args()

if args.dataset == 'M2CAI':
    args.out_features = 8
elif args.dataset == 'HeiCo':
    args.out_features = 14
else:
    args.out_features = 7


# Inicializar W&B con los argumentos como configuración
current_time = time.localtime()

wandb.init(
    project="Transvnet-COLAS 2025",
    entity='endovis_bcv', 
    name=time.strftime("%Y-%m-%d %H:%M:%S", current_time),
    config=vars(args) # Guarda todos los argumentos en wandb.config
)

dataset = args.dataset

num_videos_dataset = {'Autolaparo': [10, 4],
                      'Cholec80': [40, 40],
                      'HeiChole': [16, 8],
                      'HeiCo': [21, 9],
                      'M2CAI': [27, 14]}


def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    # train_paths_19 = train_test_paths_labels[0]
    train_paths_80 = train_test_paths_labels[0]
    val_paths_80 = train_test_paths_labels[1]
    # train_labels_19 = train_test_paths_labels[3]
    train_labels_80 = train_test_paths_labels[2]
    val_labels_80 = train_test_paths_labels[3]
    # train_num_each_19 = train_test_paths_labels[6]
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

    # train_labels_19 = np.asarray(train_labels_19, dtype=np.int64)
    train_labels_80 = np.asarray(train_labels_80, dtype=np.int64)
    val_labels_80 = np.asarray(val_labels_80, dtype=np.int64)

    train_start_vidx = []
    count = 0
    for i in range(len(train_num_each_80)):
        train_start_vidx.append(count)
        count += train_num_each_80[i]

    val_start_vidx = []
    count = 0
    for i in range(len(val_num_each_80)):
        val_start_vidx.append(count)
        count += val_num_each_80[i]

    return train_labels_80, train_num_each_80, train_start_vidx, val_labels_80, val_num_each_80, val_start_vidx


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

train_labels_80, train_num_each_80, train_start_vidx,\
    val_labels_80, val_num_each_80, val_start_vidx = get_data(f'pkl_datasets_files/train_val_paths_labels_{dataset}.pkl')


class Transformer(nn.Module):
    def __init__(self, mstcn_f_maps, mstcn_f_dim, out_features, len_q):
        super(Transformer, self).__init__()
        self.num_f_maps = mstcn_f_maps  # 32
        self.dim = mstcn_f_dim  # 2048
        self.num_classes = out_features  # 7
        self.len_q = len_q


        self.transformer = Transformer2_3_1(d_model=out_features, d_ff=mstcn_f_maps, d_k=mstcn_f_maps,
                                        d_v=mstcn_f_maps, n_layers=1, n_heads=8, len_q = len_q)
        self.fc = nn.Linear(mstcn_f_dim, out_features, bias=False)


    def forward(self, x, long_feature):
        out_features = x.transpose(1,2)
        inputs = []
        for i in range(out_features.size(1)):
            if i<self.len_q-1:
                input = torch.zeros((1, self.len_q-1-i, self.num_classes)).cuda()
                input = torch.cat([input, out_features[:, 0:i+1]], dim=1)
            else:
                input = out_features[:, i-self.len_q+1:i+1]
            inputs.append(input)
        inputs = torch.stack(inputs, dim=0).squeeze(1)
        feas = torch.tanh(self.fc(long_feature).transpose(0,1))
        output = self.transformer(inputs, feas)

        return output



with open(f"LFB/{args.dataset}/g_LFB50_train.pkl", 'rb') as f:
    g_LFB_train = pickle.load(f)

with open(f"LFB/{args.dataset}/g_LFB50_val.pkl", 'rb') as f:
    g_LFB_val = pickle.load(f)


print("load completed")

print("g_LFB_train shape:", g_LFB_train.shape)
print("g_LFB_val shape:", g_LFB_val.shape)


criterion_phase1 = nn.CrossEntropyLoss()

model = mstcn.MultiStageModel(args.mstcn_stages, args.mstcn_layers, args.mstcn_f_maps,
                            args.mstcn_f_dim, args.out_features, args.mstcn_causal_conv)


model_path = f'TeCNO50_models/{args.dataset}/'
model_name = 'TeCNO50_best_model.pth'

model.load_state_dict(torch.load(model_path+model_name))
model.cuda()
model.eval()

model1 = Transformer(args.mstcn_f_maps, args.mstcn_f_dim, args.out_features, args.sequence_length)
model1.cuda()

optimizer1 = optim.Adam(model1.parameters(), lr=args.learning_rate)

best_model_wts = copy.deepcopy(model1.state_dict())
best_val_fscore_phase = 0.0
correspond_train_acc_phase = 0.0
best_epoch = 0

train_we_use_start_idx_80 = [x for x in range(num_videos_dataset[dataset][0])]
val_we_use_start_idx_80 = [x for x in range(num_videos_dataset[dataset][1])]

#Information used for saving the id of each prediction and perform relaxed evaluation,
#  its added to the video_id used in validation loop
gap_for_dataset = {'Cholec80': 62,
                   'M2CAI': 183}

for epoch in tqdm(range(args.max_epochs)):

    torch.cuda.empty_cache()
    random.shuffle(train_we_use_start_idx_80)
    train_idx_80 = []
    model1.train()
    train_loss_phase = 0.0
    train_corrects_phase = 0
    batch_progress = 0.0
    running_loss_phase = 0.0
    minibatch_correct_phase = 0.0

    for i in tqdm(train_we_use_start_idx_80):

        optimizer1.zero_grad()
        labels_phase = []
        for j in range(train_start_vidx[i], train_start_vidx[i]+train_num_each_80[i]):
            labels_phase.append(train_labels_80[j])
        labels_phase = torch.LongTensor(labels_phase)
        if use_gpu:
            labels_phase = labels_phase.to(device)
        else:
            labels_phase = labels_phase

        long_feature = get_long_feature(start_index=train_start_vidx[i],
                                        lfb=g_LFB_train, LFB_length=train_num_each_80[i])

        long_feature = (torch.Tensor(long_feature)).to(device)
        video_fe = long_feature.transpose(2, 1)


        out_features = model.forward(video_fe)[-1]
        out_features = out_features.squeeze(1)
        p_classes1 = model1(out_features.detach(), long_feature)


        p_classes1 = p_classes1.squeeze()
        clc_loss = criterion_phase1(p_classes1, labels_phase)

        _, preds_phase = torch.max(p_classes1.data, 1)

        loss = clc_loss

        loss.backward()
        optimizer1.step()

        running_loss_phase += clc_loss.data.item()
        train_loss_phase += clc_loss.data.item()

        batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
        train_corrects_phase += batch_corrects_phase
        minibatch_correct_phase += batch_corrects_phase

        wandb.log({"Train_loss": clc_loss.data.item()})
        

    train_accuracy_phase = float(train_corrects_phase) / len(train_labels_80)
    train_average_loss_phase = train_loss_phase

    # Sets the module in evaluation mode.
    model.eval()
    model1.eval()
    val_loss_phase = 0.0
    val_corrects_phase = 0
    val_progress = 0
    val_all_preds_phase = []
    val_all_labels_phase = []
    val_acc_each_video = []
    video_names = []

    with torch.no_grad():

        for i in tqdm(val_we_use_start_idx_80):

            labels_phase = []
            for j in range(val_start_vidx[i], val_start_vidx[i] + val_num_each_80[i]):
                labels_phase.append(val_labels_80[j])
            labels_phase = torch.LongTensor(labels_phase)
            if use_gpu:
                labels_phase = labels_phase.to(device)
            else:
                labels_phase = labels_phase

            long_feature = get_long_feature(start_index=val_start_vidx[i],
                                            lfb=g_LFB_val, LFB_length=val_num_each_80[i])

            long_feature = (torch.Tensor(long_feature)).to(device)
            video_fe = long_feature.transpose(2, 1)

            out_features = model.forward(video_fe)[-1]
            out_features = out_features.squeeze(1)
            p_classes1 = model1(out_features, long_feature)

            p_classes = p_classes1.squeeze()
            clc_loss = criterion_phase1(p_classes, labels_phase)

            _, preds_phase = torch.max(p_classes.data, 1)
            loss_phase = criterion_phase1(p_classes, labels_phase)

            val_loss_phase += loss_phase.data.item()

            val_corrects_phase += torch.sum(preds_phase == labels_phase.data)
            val_acc_each_video.append(float(torch.sum(preds_phase == labels_phase.data))/val_num_each_80[i])
            # TODO

            for j in range(len(preds_phase)):
                val_all_preds_phase.append(int(preds_phase.data.cpu()[j]))
            for j in range(len(labels_phase)):
                val_all_labels_phase.append(int(labels_phase.data.cpu()[j]))

            if args.dataset == 'Cholec80' or args.dataset == 'M2CAI':
                for _ in range(len(preds_phase)):
                    video_names.append('video_{:02d}'.format(gap_for_dataset[args.dataset] + i))


    #evaluation only for training reference
    val_accuracy_phase = float(val_corrects_phase) / len(val_labels_80)
    val_acc_video = np.mean(val_acc_each_video)
    val_average_loss_phase = val_loss_phase

    val_recall_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average='macro')
    val_precision_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average='macro')
    val_jaccard_phase = metrics.jaccard_score(val_all_labels_phase, val_all_preds_phase, average='macro')
    val_fscore_phase = metrics.f1_score(val_all_labels_phase, val_all_preds_phase, average='macro')
    val_precision_each_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average=None)
    val_recall_each_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average=None)

    wandb.log({'Accuracy': val_accuracy_phase})
    wandb.log({'Recall': val_recall_phase})
    wandb.log({'Precision': val_precision_phase})
    wandb.log({'Jaccard': val_jaccard_phase})
    wandb.log({'Fscore': val_fscore_phase})


    results = None
    # Relaxed Evaluation if its neccesary
    if args.dataset == 'Cholec80':
        results = relaxed_evaluation_Cholec80(val_all_preds_phase, val_all_labels_phase, video_names)
        wandb.log({args.dataset: results[args.dataset]})
    
    if args.dataset == 'M2CAI':
        results = relaxed_evaluation_M2CAI(val_all_preds_phase, val_all_labels_phase, video_names)
        wandb.log({args.dataset: results[args.dataset]})

    if val_fscore_phase > best_val_fscore_phase:
        best_val_fscore_phase = val_fscore_phase
        best_model_wts = copy.deepcopy(model1.state_dict())
        best_epoch = epoch

        print("best_epoch", str(best_epoch))

        base_name = f"Transvnet_best_model_epoch_{epoch}_f1_{val_fscore_phase}"

        # Save just the best model based on F1 score
        os.makedirs(f"Transvnet_models/{args.dataset}", exist_ok=True)        
        torch.save(best_model_wts, f"Transvnet_models/{args.dataset}/{base_name}.pth")
        

        if args.dataset == 'Cholec80' or args.dataset == 'M2CAI':
            wandb.log({'Best metrics': results[args.dataset]})

        else:
            results = {'Accuracy': val_accuracy_phase, 'Recall': val_recall_phase, 'Precision': val_precision_phase,
                       'Jaccard': val_jaccard_phase, 'F1_score': val_fscore_phase}
            
            wandb.log({'Best Metrics': results})



