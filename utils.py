import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataloader.dataloader import NPZDataset, train_collate_fn
import numpy as np
from tqdm import tqdm
import time
import torch.distributed as dist
class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = self.kernel_size
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        if(out_dim%2!=0):
            out_dim+=1
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.max_pool(net)
        
        
        return net
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do
        
        # 使用动态 LayerNorm
        self.rms1 = LayerNorm(data_format="channels_first")
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # 第二个卷积层
        self.rms2 = LayerNorm(data_format="channels_first")
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        out = x
        # the first conv
        if not self.is_first_block:
            if self.use_bn:
                out = self.rms1(out)
        out = self.relu1(out)
        if self.use_do:
            out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.rms2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        if out.shape[-1]!=identity.shape[-1]:
            out=F.pad(out, (0, identity.shape[-1]-out.shape[-1]), "constant", 0)

        # shortcut
        out += identity

        return out
class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples,target_length)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups=1, n_block=4, downsample_gap=2, increasefilter_gap=10, use_bn=True, use_do=True, verbose=False, configs=None):
        super(ResNet1D, self).__init__()
        
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.configs = configs
        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base mode
        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1,groups=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            downsample = True
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            if i_block==0:
                tmp_block = BasicBlock(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=self.kernel_size, 
                    stride = self.stride, 
                    groups = self.groups, 
                    downsample=True, 
                    use_bn = self.use_bn, 
                    use_do = self.use_do, 
                    is_first_block=is_first_block)
            else:
                if self.configs.channel!='Resp':
                        tmp_block = BasicBlock(
                            in_channels=in_channels, 
                            out_channels=out_channels, 
                            kernel_size=8, 
                            stride=3, 
                            groups = self.groups, 
                            downsample=downsample, 
                            use_bn = self.use_bn, 
                            use_do = self.use_do, 
                            is_first_block=is_first_block)
                else:
                        tmp_block = BasicBlock(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        kernel_size=4, 
                        stride=2, 
                        groups = self.groups, 
                        downsample=downsample, 
                        use_bn = self.use_bn, 
                        use_do = self.use_do, 
                        is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        
        
    def forward(self, x):
        
        out = x
        
        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)   
        return out
class NPZDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, configs, mode='train'):
        self.file_paths = file_paths
        self.channel = configs.channel
        self.mode = mode
        self.jitter_ratio = configs.augmentation.jitter_ratio
        
        # Store each segment index mapping and metadata
        self.segment_to_file = []  # Used to map segment index to file index
        self.metadata = []
        
        # Cache management
        self.cache_size = 10000  # Maximum cache file number
        self.cache = {}  # File data cache
        self.file_access_count = {}  # Track file access times for LRU replacement
        self.access_counter = 0  # Global access counter
        
        # Preprocessing: scan all files to build index
        for file_idx, file_path in enumerate(tqdm(file_paths, desc="Reading metadata")):
            try:
                # Only read metadata, not load all content
                data = torch.load(file_path, map_location='cpu',mmap=True)
                # Select data based on channel
                if self.channel == 'ECG':
                    x_shape = data['x'].shape
                elif self.channel == 'EEG':
                    x_shape = data['x1'].shape
                elif self.channel == 'EEG2':
                    x_shape = data['x2'].shape
                elif self.channel == 'Resp':
                    x_shape = data['x3'].shape
                
                # Check x shape, ensure no 0 dimension and is 2D tensor [num_segments, signal_length]
                if 0 not in x_shape and len(x_shape) == 2:
                    num_segments = x_shape[0]
                    print(file_idx,num_segments)
                    # For each segment, create a mapping and metadata entry
                    for seg_idx in range(num_segments):
                        self.segment_to_file.append((file_idx, seg_idx))
                        if self.mode == 'train':
                            self.metadata.append({
                                'y1': data['y1'],
                                'y2': data['y2'],
                                'y3': data['y3'],
                                'y6': data['y6'],
                            })
                        else:
                                self.metadata.append({
                                    'y1': data['y1'],
                                    'y2': data['y2'],
                                    'y3': data['y3'],
                                    'age': data['age'],
                                    'sex': data['sex'],
                                    'race': data['race'],
                                    'bmi': data['bmi'],
                                })
                else:
                    print(f"Skip {file_path}, because x shape is not valid: {x_shape}")
                    
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        
        print(f"Loaded {len(self.segment_to_file)} segments")
        
        # Preload some data to speed up
        self._preload_data()
    def _preload_data(self):
        """Preload some data to cache"""
        print("Preloading data to cache...")
        for file_idx in tqdm(range(min(len(self.file_paths), self.cache_size))):
            if file_idx not in self.cache:
                file_path = self.file_paths[file_idx]
                print(file_path)
                data = torch.load(file_path, map_location=torch.device('cpu'), mmap=True)
                if self.channel == 'ECG':
                    x = data['x']
                elif self.channel == 'EEG':
                    x = data['x1']
                elif self.channel == 'EEG2':
                    x = data['x2']
                elif self.channel == 'Resp':
                    x = data['x3']
                # labels = data['labels']
                labels = data['labels']
                # y6 = data['y6']
                self.cache[file_idx] = x,labels
    
    def _get_data(self, file_idx):
        """Get data from cache or file"""
        if file_idx in self.cache:
            return self.cache[file_idx]
        
        # If not in cache, load data
        file_path = self.file_paths[file_idx]
        data = torch.load(file_path, map_location=torch.device('cpu'), mmap=True)
        if self.channel == 'ECG':
            x = data['x']
        elif self.channel == 'EEG':
            x = data['x1']
        elif self.channel == 'EEG2':
            x = data['x2']
        elif self.channel == 'Resp':
            x = data['x3']
        # labels = data['labels']
        labels = data['labels']
        # Update cache
        if len(self.cache) >= self.cache_size:
            # Randomly delete a cache item
            remove_key = list(self.cache.keys())[0]
            del self.cache[remove_key]
        self.cache[file_idx] = x,labels
        
        return x,labels
    
    def __len__(self):
        return len(self.flattened_indices)
    
    def __len__(self):
        return len(self.segment_to_file)
    
    def __getitem__(self, idx):
        # Get corresponding file and segment index
        file_idx, seg_idx = self.segment_to_file[idx]
        # meta = self.metadata[idx]
        
        # Get data from cache, if not in cache, load data
        x,labels = self._get_data(file_idx)
        # Get specified segment
        x = x[seg_idx:seg_idx+1]  # Keep 2D shape [1, signal_length]
        labels = labels[seg_idx:seg_idx+1]  # Ensure label dimension matches
        
        
        # Convert to tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        else:
            x = x.float()
        # If training mode, add jitter as data augmentation
        if self.mode == 'train':
            x = x + self.jitter_ratio * torch.randn(x.size())
            # y = torch.tensor(meta['y']).long()
            labels = labels.long()  # Directly convert type, no need to re-create tensor
            y1 = torch.tensor(self.metadata[idx]['y1']).long()
            y2 = torch.tensor(self.metadata[idx]['y2']).long()
            y3 = torch.tensor(self.metadata[idx]['y3']).long()
            y6 = torch.tensor(self.metadata[idx]['y6']).long()
            return x, labels,y6,None,None,None,None,y1,y2,y3
        else:
            y1 = torch.tensor(self.metadata[idx]['y1']).long()
            y2 = torch.tensor(self.metadata[idx]['y2']).long()
            y3 = torch.tensor(self.metadata[idx]['y3']).long()
            age = torch.tensor(self.metadata[idx]['age']).clone().detach().float()
            sex = torch.tensor(self.metadata[idx]['sex']).clone().detach().long()
            race = torch.tensor(self.metadata[idx]['race']).clone().detach().long()
            bmi = torch.tensor(self.metadata[idx]['bmi']).clone().detach().float()
            labels = labels.long()  # Directly convert type, no need to re-create tensor
            return x, labels,age,sex,race,bmi,y1,y2,y3 
def data_generator_for_wholenight(data_path, configs):
    # Create save directory list instead of loading all data
    train_files = [os.path.join(data_path+"/train", f) for f in os.listdir(data_path+"/train") if f.endswith('.pt')]
    test_files = [os.path.join(data_path+"/test", f) for f in os.listdir(data_path+"/test") if f.endswith('.pt')]
    # Create dataset
    train_dataset = NPZDataset(train_files, configs, mode='train')
    valid_dataset = NPZDataset(test_files, configs, mode='train')
    
    # Determine number of workers to use, if not specified, use default value
    num_workers = getattr(configs, 'num_workers', 4)
    prefetch_factor = 2  # Pre-fetch factor, number of batches to pre-fetch for each worker
    train_loader = DataLoader(
        train_dataset,
        batch_size=configs.batch_size1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_collate_fn,  # Use named function instead of lambda
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=configs.batch_size1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_collate_fn,  # Use named function instead of lambda
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    # Return training and validation data loaders, test set set to None (because there is no separate test set)
    return train_loader, valid_loader
def train_epoch(model, train_loader, optimizer, train_mode, device, 
                mask_ratio=0.9, num_masked=20, scheduler=None, scaler=None, 
                accumulation_steps=1, epoch=0):
    model.train()
    epoch_loss = 0
    epoch_contrastive_loss = 0
    epoch_constrain_loss = 0
    epoch_correct = 0
    epoch_total = 0
    start = time.time()
    train_loader = tqdm(train_loader, total=len(train_loader), unit="batch")
    
    for batch_idx, data in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        model.zero_grad()
        x = data[0].float().to(device)
        y = data[1].long().to(device)
        mask=y<=4
        x=x[mask]
        y=y[mask]
        print(f"(x) - mean: {torch.mean(x):.3f}, std: {torch.std(x):.3f}")
        if scaler is not None:
            # Half precision training
            with torch.cuda.amp.autocast():
                if train_mode in ['supervised', 'linear_prob', 'finetune']:
              
                    loss, pred = model.supervised_train_forward(x, y) 
                    
                    # Calculate accuracy
                    _, predict_targets = torch.max(pred.detach().data, 1)
                    correct = (predict_targets == y).sum().item()
                    total = y.size(0)
                    
                    batch_loss = loss.item()
                    
                    # Accumulate statistics
                    epoch_loss += batch_loss
                    epoch_correct += correct
                    epoch_total += total
                    
                    # Update progress bar
                    train_loader.set_postfix(
                        loss=f"{epoch_loss/batch_idx:.4f}",
                        acc=f"{epoch_correct/epoch_total:.4f}" if epoch_total > 0 else "N/A",
                        time=time.time()-start,
                        epoch=epoch
                    )
                    
                else:  # SSL training
                    loss, (contrastive_loss, constrain_loss) = model.ssl_train_forward(x, mask_ratio, num_masked)
                    batch_loss = loss.item()
                    batch_contrastive_loss = contrastive_loss
                    batch_constrain_loss = constrain_loss
                    
                    # Accumulate statistics
                    epoch_loss += batch_loss
                    epoch_contrastive_loss += batch_contrastive_loss
                    epoch_constrain_loss += batch_constrain_loss
                    
                    # Update progress bar
                    train_loader.set_postfix(
                        loss=f"{epoch_loss/batch_idx:.4f}",
                        contrastive=f"{epoch_contrastive_loss/batch_idx:.4f}",
                        constrain=f"{epoch_constrain_loss/batch_idx:.4f}",
                        time=time.time()-start,
                        epoch=epoch
                    )
                
                # Scale loss to adapt to gradient accumulation
                loss = loss / accumulation_steps
            
            # Scale gradient and backpropagate
            scaler.scale(loss).backward()
            
            # Optimize every accumulation_steps steps
            if (batch_idx % accumulation_steps == 0) or (batch_idx == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()
        else:
            # Full precision training
            if train_mode in ['supervised', 'linear_prob', 'finetune']:
                loss, pred = model(x, y) 
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy for current batch
                _, predict_targets = torch.max(pred.detach().data, 1)
                correct = (predict_targets == y.squeeze()).sum().item()
                total = y.size(0)
                batch_loss = loss.item()
                # Accumulate statistics
                epoch_loss += batch_loss
                epoch_correct += correct
                epoch_total += total
                
                # Update progress bar
                train_loader.set_postfix(
                    loss=f"{epoch_loss/batch_idx:.4f}",
                    acc=f"{epoch_correct/epoch_total:.4f}" if epoch_total > 0 else "N/A",
                    time=time.time()-start,
                    epoch=epoch
                )
            
            else:  # SSL training
                loss, (contrastive_loss, constrain_loss) = model.ssl_train_forward(x, mask_ratio, num_masked)
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item()
                batch_contrastive_loss = contrastive_loss
                batch_constrain_loss = constrain_loss
                
                # Accumulate statistics
                epoch_loss += batch_loss
                epoch_contrastive_loss += batch_contrastive_loss
                epoch_constrain_loss += batch_constrain_loss
                
                # Update progress bar
                train_loader.set_postfix(
                    loss=f"{epoch_loss/batch_idx:.4f}",
                    contrastive=f"{epoch_contrastive_loss/batch_idx:.4f}",
                    constrain=f"{epoch_constrain_loss/batch_idx:.4f}",
                    time=time.time()-start,
                    epoch=epoch
                )
            
            if scheduler is not None:
                scheduler.step()
    
    # Calculate average loss and accuracy
    if train_mode in ['supervised', 'linear_prob', 'finetune']:
        epoch_loss /= len(train_loader)
        epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
        return epoch_loss, epoch_acc
    else:
        epoch_loss /= len(train_loader)
        epoch_contrastive_loss /= len(train_loader)
        epoch_constrain_loss /= len(train_loader)
        return epoch_loss, (epoch_contrastive_loss, epoch_constrain_loss)
