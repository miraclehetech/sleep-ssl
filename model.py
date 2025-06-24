from utils import ResNet1D, posemb_sincos_1d
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch
from loss import Similarity_Loss, TotalCodingRate
from focal_loss import FocalLoss
from einops import rearrange
from model import Transformer

class MTS_LOF_revised(nn.Module):
    def __init__(self, configs):
        super(MTS_LOF_revised, self).__init__()
        
        if configs.channel=='Resp':
            self.conv_block = ResNet1D(
                in_channels=configs.input_channels, 
                base_filters=configs.embed_dim//2,
                kernel_size=configs.kernel_size,
                stride=configs.stride,
                configs=configs,
                groups=1,  # 减少分组，增加特征交互
                n_block=4,  # 增加网络深度
                downsample_gap=1,
                increasefilter_gap=3,
                use_bn=True,
                use_do=True,
        )
        else:
            self.conv_block = ResNet1D(
                in_channels=configs.input_channels, 
                base_filters=configs.embed_dim//2,
                kernel_size=configs.kernel_size,
                stride=configs.stride,
                configs=configs,
                groups=1,  # 减少分组，增加特征交互
                n_block=4,  # 增加网络深度
                downsample_gap=1,
                increasefilter_gap=3,
                use_bn=True,
                use_do=True,
        )
        self.transformer_encoder = Transformer(
            configs.embed_dim, 
            depth=6,  # 减少Transformer层数
            heads=8, 
            dim_head=configs.embed_dim//8, 
            mlp_dim=configs.embed_dim*4
        )
        
        self.linear = nn.Linear(configs.embed_dim, configs.num_classes)
        
        self.inv_loss = Similarity_Loss()
        self.tcr_loss = TotalCodingRate(eps=0.4)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, configs.embed_dim))
        self.decoder = Transformer(
            configs.embed_dim, 
            depth=4,  # 减少decoder层数
            heads=8, 
            dim_head=configs.embed_dim//8, 
            mlp_dim=configs.embed_dim*4
        )
        self.sample_rate = 125
        self.attention_weights = nn.Sequential(
            nn.LayerNorm(configs.embed_dim),  # 添加归一化
            nn.Linear(configs.embed_dim, configs.embed_dim//2),
            nn.GELU(),  # 使用 GELU 替代 Tanh
            nn.Dropout(0.1),  # 添加 dropout
            nn.Linear(configs.embed_dim//2, 1),
        )
        self.focal_loss = FocalLoss(gamma=2.0)
    def forward(self, x_in):
        batch_size,dim,length=x_in.shape
        x_in=self.conv_block(x_in)
        embed_dim=x_in.shape[1]
        x_in=x_in.permute(0,2,1)
        # 添加残差连接
        x_in=x_in.reshape(batch_size,-1,embed_dim)
        b, n, _ = x_in.shape
        pe = posemb_sincos_1d(x_in)
        x_in = rearrange(x_in, 'b ... d -> b (...) d') + pe
        x_in = self.transformer_encoder(x_in).mean(dim=1)
        rep=x_in.detach()
        return self.linear(x_in), rep
    def ssl_train_forward(self, x, mask_ratio=0.75, num_masked=20):
        batch_size,dim,length=x.shape
        x_in=self.conv_block(x)
        embed_dim=x_in.shape[1]
        x_in=x_in.permute(0,2,1)
        x_in=x_in.reshape(batch_size,-1,embed_dim)
        print(x_in.dtype)
        print(x_in.shape)
        b, n, _ = x_in.shape
        pe = posemb_sincos_1d(x_in)
        x_in = rearrange(x_in, 'b ... d -> b (...) d') + pe
        z_avg = self.transformer_encoder(x_in).mean(dim=1)
        z_avg = F.normalize(z_avg, p=2)
        z_list = []

        for _ in range(num_masked):
            z, mask, ids_restore = self.random_masking(x_in, mask_ratio)
            z = self.transformer_encoder(z)
            mask_tokens = self.mask_token.repeat(z.shape[0], ids_restore.shape[1] - z.shape[1], 1)
            z = torch.cat([z, mask_tokens], dim=1)
            z = torch.gather(z, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, z.shape[2]))

            pe = posemb_sincos_1d(z)
            z = rearrange(z, 'b ... d -> b (...) d') + pe
            z = self.decoder(z).mean(dim=1)
            z = F.normalize(z, p=2)
            z_list.append(z)

        contrastive_loss = 100 * self.inv_loss(z_list, z_avg)
        diversity_loss = 1* self.tcr_loss(z_list)
        # push_loss = 0.0 * self.push_away_loss(z_avg)
        loss = contrastive_loss + diversity_loss 
        return loss, [contrastive_loss.item(), diversity_loss.item()]

    def supervised_train_forward(self, x, y):
        pred, _ = self.forward(x)
        criterion=nn.CrossEntropyLoss()
        loss=criterion(pred,y)
        return loss, pred.detach()

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def push_away_loss(self, features):
        """计算特征之间的排斥力"""
        sim_matrix = torch.matmul(features, features.t())
        mask = torch.eye(features.shape[0], device=features.device)
        sim_matrix = sim_matrix * (1 - mask)  # 排除自身相似度
        return sim_matrix.mean()
        return loss, [contrastive_loss.item(), diversity_loss.item(), push_loss.item()]

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore