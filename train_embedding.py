#some code adapted from https://github.com/YuemingJin/MTRCNet-CL
# and https://github.com/YuemingJin/TMRNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import os
import time
import pickle
import numpy as np
from torchvision.transforms import Lambda
import argparse
import copy
import random
import numbers
from sklearn import metrics
import wandb
from datetime import datetime
from tqdm import tqdm


# Seed everything
seed = 19980125

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)





parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('-g', '--gpu', default=True, type=bool, help='gpu use, default True')
parser.add_argument('-s', '--seq', default=1, type=int, help='sequence length, default 10')
parser.add_argument('-t', '--train', default=400, type=int, help='train batch size, default 400')
parser.add_argument('-v', '--val', default=400, type=int, help='valid batch size, default 10')
parser.add_argument('-o', '--opt', default=0, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=25, type=int, help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=8, type=int, help='num of workers to use, default 4')
parser.add_argument('-f', '--flip', default=1, type=int, help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('-l', '--lr', default=5e-4, type=float, help='learning rate for optimizer, default 5e-5')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=5e-4, type=float, help='weight decay for sgd, default 0')
parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
parser.add_argument('--sgdstep', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')
parser.add_argument('--dataset', default='Cholec80', type=str)

args = parser.parse_args()

gpu_usg = args.gpu
sequence_length = args.seq
train_batch_size = args.train
val_batch_size = args.val
optimizer_choice = args.opt
multi_optim = args.multi
epochs = args.epo
workers = args.work
use_flip = args.flip
crop_type = args.crop
learning_rate = args.lr
momentum = args.momentum
weight_decay = args.weightdecay
dampening = args.dampening
use_nesterov = args.nesterov

sgd_adjust_lr = args.sgdadjust
sgd_step = args.sgdstep
sgd_gamma = args.sgdgamma

# Configure number of classes depending on dataset
if args.dataset == 'HeiCo':
    args.num_classes = 14

elif args.dataset == 'M2CAI':
    args.num_classes = 8

else:
    args.num_classes = 7



num_gpu = torch.cuda.device_count()
use_gpu = (torch.cuda.is_available() and gpu_usg)
device = torch.device("cuda:0" if use_gpu else "cpu")

# Configure wandb
exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

wandb.init(
    project='Resnet_encoder-COLAS2025', 
    entity = 'endovis_bcv',
    config=vars(args), 
    name=exp_name
    )


print('number of gpu   : {:6d}'.format(num_gpu))
print('sequence length : {:6d}'.format(sequence_length))
print('train batch size: {:6d}'.format(train_batch_size))
print('valid batch size: {:6d}'.format(val_batch_size))
print('optimizer choice: {:6d}'.format(optimizer_choice))
print('multiple optim  : {:6d}'.format(multi_optim))
print('num of epochs   : {:6d}'.format(epochs))
print('num of workers  : {:6d}'.format(workers))
print('test crop type  : {:6d}'.format(crop_type))
print('whether to flip : {:6d}'.format(use_flip))
print('learning rate   : {:.4f}'.format(learning_rate))
print('momentum for sgd: {:.4f}'.format(momentum))
print('weight decay    : {:.4f}'.format(weight_decay))
print('dampening       : {:.4f}'.format(dampening))
print('use nesterov    : {:6d}'.format(use_nesterov))
print('method for sgd  : {:6d}'.format(sgd_adjust_lr))
print('step for sgd    : {:6d}'.format(sgd_step))
print('gamma for sgd   : {:.4f}'.format(sgd_gamma))


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class RandomCrop(object):

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.count = 0

    def __call__(self, img):

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        random.seed(self.count // sequence_length)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # print(self.count, x1, y1)
        self.count += 1
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontalFlip(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        prob = random.random()
        self.count += 1
        # print(self.count, seed, prob)
        if prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        self.count += 1
        angle = random.randint(-self.degrees, self.degrees)
        return TF.rotate(img, angle)


class ColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        self.count += 1
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(- self.hue, self.hue)

        img_ = TF.adjust_brightness(img, brightness_factor)
        img_ = TF.adjust_contrast(img_, contrast_factor)
        img_ = TF.adjust_saturation(img_, saturation_factor)
        img_ = TF.adjust_hue(img_, hue_factor)

        return img_


class CholecDataset(Dataset):
    def __init__(self, file_paths, file_labels, transform=None,
                 loader=pil_loader):
        
        self.file_paths = file_paths
        self.file_labels_phase = file_labels
        self.transform = transform
        self.loader = loader
        self.images_base_path = os.path.join('DATASETS', 'PHASES', 'frames')

    def __getitem__(self, index):
        img_names = os.path.join(self.images_base_path, self.file_paths[index])
        labels_phase = self.file_labels_phase[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_phase

    def __len__(self):
        return len(self.file_paths)


class resnet_lstm(torch.nn.Module):
    def __init__(self, num_classes):
        super(resnet_lstm, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Linear(512, num_classes))


    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        y = self.fc(x)
        return y


def get_useful_start_idx(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx


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


    # train_labels_19 = np.asarray(train_labels_19, dtype=np.int64)
    train_labels_80 = np.asarray(train_labels_80, dtype=np.int64)
    val_labels_80 = np.asarray(val_labels_80, dtype=np.int64)


    train_transforms = None
    test_transforms = None

    if use_flip == 0:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(224),
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif use_flip == 1:
        #Entramos aca y eso es lo mismo que reportan en el paper
        # data augmentation is applied, including 224×224 cropping, random mirroring, and color jittering.

        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(224),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            RandomHorizontalFlip(),
            RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])

    if crop_type == 0:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif crop_type == 1:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif crop_type == 2:
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif crop_type == 5:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.FiveCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])(crop)
                     for crop in crops]))
        ])
    elif crop_type == 10:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])(crop)
                     for crop in crops]))
        ])

    # train_dataset_19 = CholecDataset(train_paths_19, train_labels_19, train_transforms)
    train_dataset_80 = CholecDataset(train_paths_80, train_labels_80, train_transforms)
    val_dataset_80 = CholecDataset(val_paths_80, val_labels_80, test_transforms)


    return train_dataset_80, train_num_each_80, val_dataset_80, val_num_each_80



class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


sig_f = nn.Sigmoid()


# Deleted Val mini batch function


def train_model(train_dataset, train_num_each, val_dataset, val_num_each, args):

    train_dataset_80, \
    train_num_each_80, \
    val_dataset, \
    val_num_each = train_dataset, train_num_each, val_dataset, val_num_each

    # train_useful_start_idx_19 = get_useful_start_idx(sequence_length, train_num_each_19)
    train_useful_start_idx_80 = get_useful_start_idx(sequence_length, train_num_each_80)
    val_useful_start_idx = get_useful_start_idx(sequence_length, val_num_each)


    num_train_we_use_80 = len(train_useful_start_idx_80)
    num_val_we_use = len(val_useful_start_idx)


    train_we_use_start_idx_80 = train_useful_start_idx_80
    val_we_use_start_idx = val_useful_start_idx


    train_idx = []
    for i in range(num_train_we_use_80):
        for j in range(sequence_length):
            train_idx.append(train_we_use_start_idx_80[i] + j)

    val_idx = []
    for i in range(num_val_we_use):
        for j in range(sequence_length):
            val_idx.append(val_we_use_start_idx[i] + j)


    num_train_all = len(train_idx)
    num_val_all = len(val_idx)

    # print('num train start idx 19: {:6d}'.format(len(train_useful_start_idx_19)))
    print('num train start idx 80: {:6d}'.format(len(train_useful_start_idx_80)))
    print('num of all train use: {:6d}'.format(num_train_all))
    print('num of all valid use: {:6d}'.format(num_val_all))

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=SeqSampler(val_dataset, val_idx),
        num_workers=workers,
        pin_memory=False
    )


    model = resnet_lstm(args.num_classes)

    if use_gpu:
        model = DataParallel(model)
        model.to(device)

    
    criterion_phase = nn.CrossEntropyLoss(size_average=False)
    optimizer = None
    exp_lr_scheduler = None

    if multi_optim == 0:
        if optimizer_choice == 0:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=dampening,
                                  weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif multi_optim == 1:
        if optimizer_choice == 0:
            optimizer = optim.SGD([
                {'params': model.module.share.parameters()},
                #{'params': model.module.lstm.parameters(), 'lr': learning_rate},
                {'params': model.module.fc.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam([
                {'params': model.module.share.parameters()},
                #{'params': model.module.lstm.parameters(), 'lr': learning_rate},
                {'params': model.module.fc.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10)

    best_model_wts = copy.deepcopy(model.module.state_dict())
    best_val_f1_phase = 0.0
    best_epoch = 0

    for epoch in tqdm(range(epochs)):

        torch.cuda.empty_cache()
        np.random.shuffle(train_we_use_start_idx_80)
        train_idx_80 = []

        for i in range(num_train_we_use_80):
            for j in range(sequence_length):
                train_idx_80.append(train_we_use_start_idx_80[i] + j)

        train_loader_80 = DataLoader(
            train_dataset_80,
            batch_size=train_batch_size,
            sampler=SeqSampler(train_dataset_80, train_idx_80),
            num_workers=workers,
            pin_memory=False
        )

        # Sets the module in training mode.
        model.train()
        train_loss_phase = 0.0
        train_corrects_phase = 0
        running_loss_phase = 0.0
        minibatch_correct_phase = 0.0
        
        print('\n Training \n')
        for i, data in enumerate(tqdm(train_loader_80)):
            optimizer.zero_grad()
            if use_gpu:
                inputs, labels_phase = data[0].to(device), data[1].to(device)
            else:
                inputs, labels_phase = data[0], data[1]

            labels_phase = labels_phase[(sequence_length - 1)::sequence_length]

            inputs = inputs.view(-1, sequence_length, 3, 224, 224)
            outputs_phase = model.forward(inputs)
            outputs_phase = outputs_phase[sequence_length - 1::sequence_length]

            _, preds_phase = torch.max(outputs_phase.data, 1)
            loss_phase = criterion_phase(outputs_phase, labels_phase)

            loss = loss_phase
            loss.backward()
            optimizer.step()

            running_loss_phase += loss_phase.data.item()
            train_loss_phase += loss_phase.data.item()

            batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
            train_corrects_phase += batch_corrects_phase
            minibatch_correct_phase += batch_corrects_phase

            wandb.log({
                'Train loss': loss_phase.data.item(),
                'Train Accuracy': batch_corrects_phase.data.item() / preds_phase.shape[0]
                })


        # Sets the module in evaluation mode.
        model.eval()
        val_loss_phase = 0.0
        val_corrects_phase = 0
        val_all_preds_phase = []
        val_all_labels_phase = []

        print('\n Testing \n')
        with torch.no_grad():
            for data in tqdm(val_loader):
                if use_gpu:
                    inputs, labels_phase = data[0].to(device), data[1].to(device)
                else:
                    inputs, labels_phase = data[0], data[1]

                labels_phase = labels_phase[(sequence_length - 1)::sequence_length]

                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs_phase = model.forward(inputs)
                outputs_phase = outputs_phase[sequence_length - 1::sequence_length]

                _, preds_phase = torch.max(outputs_phase.data, 1)
                loss_phase = criterion_phase(outputs_phase, labels_phase)

                val_loss_phase += loss_phase.data.item()

                val_corrects_phase += torch.sum(preds_phase == labels_phase.data)
                # TODO

                for i in range(len(preds_phase)):
                    val_all_preds_phase.append(int(preds_phase.data.cpu()[i]))
                for i in range(len(labels_phase)):
                    val_all_labels_phase.append(int(labels_phase.data.cpu()[i]))



        val_accuracy_phase = float(val_corrects_phase) / float(num_val_we_use)
        val_average_loss_phase = val_loss_phase / num_val_we_use

        val_recall_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average='macro')
        val_precision_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average='macro')
        val_jaccard_phase = metrics.jaccard_score(val_all_labels_phase, val_all_preds_phase, average='macro')
        val_f1_phase = metrics.f1_score(val_all_labels_phase, val_all_preds_phase, average='macro')



        wandb.log({
            'Val accuracy': val_accuracy_phase,
            'Val precision': val_precision_phase,
            'Val recall': val_recall_phase,
            'Val F1': val_f1_phase,
            'Val Jaccard': val_jaccard_phase
            })

        if optimizer_choice == 0:
            if sgd_adjust_lr == 0:
                exp_lr_scheduler.step()
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler.step(val_average_loss_phase)

        if val_f1_phase > best_val_f1_phase:
            best_val_f1_phase = val_f1_phase
            best_model_wts = copy.deepcopy(model.module.state_dict())
            best_epoch = epoch

            os.makedirs(f'Resnet50_models/{args.dataset}', exist_ok=True)
            base_name = f"Resnet50_models/{args.dataset}/resnetfc_ce_{epoch}_f1_{best_val_f1_phase}"
            torch.save(best_model_wts, base_name + ".pth")
            print("best_epoch", str(best_epoch))



def main():
    # train_dataset_19, train_num_each_19, \
    train_dataset_80, train_num_each_80, \
    val_dataset_80, val_num_each_80  = get_data(f'pkl_datasets_files/train_val_paths_labels_{args.dataset}.pkl')
    train_model(train_dataset_80,
                train_num_each_80,
                val_dataset_80,
                val_num_each_80, args)


if __name__ == "__main__":
    main()

print('Done')

