import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler
from sklearn import model_selection
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import torch.distributed as dist
from collections import defaultdict
from scipy.stats import ttest_ind
from dataloader.dataloader import data_generator_k_fold,data_generator,data_generator_multiple_signals,data_generator_multiple_signals_external
# Set backend before using matplotlib
from lifelines.utils import concordance_index
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from frs_compute.frs import frs
# Set font, try Chinese font, if not available, fall back to system default font
from matplotlib.font_manager import fontManager
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from scipy.signal import welch
from model import MTS_LOF, MTS_LOF_revised
from utils import ResNet1D
from setup import setup
import argparse
from utils import data_generator_for_wholenight
import utils

def train_process(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use mixed precision training
    scaler = torch.amp.GradScaler(device=device) if device.type == 'cuda' else None
    
    # Configuration settings
    data_type = 'SHHS1'
    os.makedirs(f'./checkpoints/{data_type}/', exist_ok=True)
    config_module = __import__(f'config_files.{data_type}_Configs', fromlist=['Config'])
    configs = config_module.Config()

    # Set random seed
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Create model
    if configs.multiple_models:
        model_zoo = nn.ModuleDict()
        for channel in configs.channel_list:
            print(channel)
            configs.channel = channel
            if channel == 'Resp':
                configs.kernel_size = 15
                configs.stride = 2
                model_zoo[channel] = MTS_LOF_revised(configs)
            else:
                configs.kernel_size = 25
                configs.stride = 6
                model_zoo[channel] = MTS_LOF_revised(configs)
        for ch in model_zoo:
            model_zoo[ch] = model_zoo[ch].to(device)
    else:
        model = MTS_LOF_revised(configs)
        model = model.to(device)
    if args.load_para and args.train_mode == 'ssl':
        pretrained_path = f'./checkpoints/{data_type}/ssl_{SEED}_embed_dim_normalize{configs.embed_dim}{configs.channel}{configs.record}.pth'
        print(f"Loading pretrained model from {pretrained_path}")
        if os.path.exists(pretrained_path):
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in pretrained_dict.items() 
                           if k in model_dict and 'linear' not in k}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict, strict=True)
            print(f"Successfully loaded pretrained model from {pretrained_path}")
        else:
            print(f"Pretrained model {pretrained_path} does not exist")
            exit()
    if args.train_mode in ['finetune', 'linear_prob', 'infer_vector','cox','visualization']:
        if configs.mutiple_models:
            for channel in configs.channel_list:
                pretrained_path = f'./checkpoints/{data_type}/ssl_{SEED}_embed_dim_normalize{configs.embed_dim}{channel}{configs.record}.pth'
                if os.path.exists(pretrained_path):
                    pretrained_dict = torch.load(pretrained_path, map_location=device)
                    model_dict = model_zoo[channel].state_dict()
                    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'linear' not in k}
                    model_dict.update(filtered_dict)
                    model_zoo[channel].load_state_dict(model_dict, strict=True)
                    print(f"Successfully loaded pretrained model from {pretrained_path}")
                else:
                    print(f"Pretrained model {pretrained_path} does not exist")
                    exit()
        else:
            pretrained_path = f'./checkpoints/{data_type}/ssl_{SEED}_embed_dim_normalize{configs.embed_dim}{configs.channel}{configs.record}.pth'
            if os.path.exists(pretrained_path):
                pretrained_dict = torch.load(pretrained_path, map_location=device)
                model_dict = model.state_dict()
                filtered_dict = {k: v for k, v in pretrained_dict.items() 
                            if k in model_dict and 'linear' not in k}
                model_dict.update(filtered_dict)
                model.load_state_dict(model_dict, strict=True)
                print(f"Successfully loaded pretrained model from {pretrained_path}")
            else:
                print(f"Pretrained model {pretrained_path} does not exist")
                exit()
    if not configs.mutiple_models:
        if args.train_mode == 'linear_prob':
            optimizer = torch.optim.AdamW(model.linear.parameters(), lr=configs.lr, weight_decay=0.05)
        elif args.train_mode == 'finetune':
                optimizer = torch.optim.AdamW([
                    {"params": model.conv_block.parameters(), "lr": configs.lr},
                    {"params": model.transformer_encoder.parameters(), "lr": configs.lr},
                    {"params": model.linear.parameters(), "lr": configs.lr}],
                    weight_decay=0.05)
        else:
            optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=configs.lr ,
                    weight_decay=0.05,
                )
    if args.train_mode in ['supervised', 'linear_prob', 'finetune', 'ssl']:
        train_dl, valid_dl, test_dl = data_generator_for_wholenight(
        args.data_path, configs
    )
        for epoch in range(1, configs.num_epoch + 1):
            print(f'Epoch: {epoch}|{configs.num_epoch} || Seed: {SEED}')
            epoch_loss, epoch_acc = utils.train_epoch(
                model, train_dl, optimizer, args.train_mode, 
                mask_ratio=0.8, num_masked=20,
                scheduler=None,
                scaler=scaler,  # Pass scaler
                epoch=epoch,
            )
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', type=str, default='ssl', 
                       choices=['ssl', 'supervised', 'linear_prob', 'finetune'])
    parser.add_argument('--data_path', type=str, 
                       default='/mnt/d/sleep/shhs/output_npz/shhs-all_signals-30s')
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading threads')
    parser.add_argument('--load_para', type=bool, default=False, help='Load parameters')
    args = parser.parse_args()
    train_process(args)  

if __name__ == "__main__":
    main()
