import torch
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange, repeat

import ray
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import sys
import os
from torch.optim.lr_scheduler import StepLR

from .diff_scheduler import NoiseScheduler
from DiTT.preprocess.utils import mask_tensor_with_masks, text_cond, stage_cond
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AutoTokenizer, AutoModel
from DiTT.sample_vae import Vae



class diffusion_loss(nn.Module):
    def __init__(self, penalty_factor=1.0):
        super(diffusion_loss, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.penalty_factor = penalty_factor

    def forward(self, y_pred_1, y_true_1, y_pred_0, y_true_0):
        # loss_mse = self.mse(y_pred_0, y_true_0)
        # loss_mae = self.mae(y_pred_1, y_true_1) * self.penalty_factor
        loss_mse = self.mse(y_pred_1, y_true_1)
        loss_mae = self.mae(y_pred_0, y_true_0) * self.penalty_factor
        return loss_mse + loss_mae


def normal_train_diff(model,
                 dataloader,
                 lr: float = 1e-4,
                 num_epoch: int = 1400,
                 pred_type: str = 'noise',
                 diffusion_step: int = 1000,
                 device=torch.device('cuda:0'),
                 is_tqdm: bool = True,
                 is_tune: bool = False,
                 mask_nonzero_ratio= None,
                 mask_zero_ratio = None):
    """通用训练函数

    Args:
        lr (float):
        momentum (float): 动量
        max_iteration (int, optional): 训练的 iteration. Defaults to 30000.
        pred_type (str, optional): 预测的类型噪声或者 x_0. Defaults to 'noise'.
        batch_size (int, optional):  Defaults to 1024.
        diffusion_step (int, optional): 扩散步数. Defaults to 1000.
        device (_type_, optional): Defaults to torch.device('cuda:0').
        is_class_condi (bool, optional): 是否采用condition. Defaults to False.
        is_tqdm (bool, optional): 开启进度条. Defaults to True.
        is_tune (bool, optional): 是否用 ray tune. Defaults to False.
        condi_drop_rate (float, optional): 是否采用 classifier free guidance 设置 drop rate. Defaults to 0..

    Raises:
        NotImplementedError: _description_
    """

    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='quadratic'
    )

    # criterion = diffusion_loss()
    criterion = nn.MSELoss()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    if is_tqdm:
        t_epoch = tqdm(range(num_epoch), ncols=100)
    else:
        t_epoch = range(num_epoch)


    model_name = "bert-base-uncased"  # 预训练的 BERT 模型bert-base-uncased"pucpr/biobertpt-all"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_model = AutoModel.from_pretrained(model_name)



    vae_model = Vae().to(device)
    #vae_model.load_state_dict(torch.load('D:\\TCGA\mbVDiT_data\\WXS\\solid\\stad\\no_pretrain.pth'))
    vae_model.load_state_dict(torch.load('D:\\TCGA\mbVDiT_data\\WXS\\solid\\stad\\pretrain_tune.pth'))

    model.train()

    for epoch in t_epoch:
        epoch_loss = 0.
        for i, (x, x_cond, cond, age_cond, _, _, _, stage) in enumerate(dataloader): # 去掉了, celltype

            _, _, _, latent_data = vae_model(x.to(device))

            age_cond = torch.squeeze(age_cond).to(device)
            # x = x.float().to(device)
            # celltype = celltype.to(device)
            # x, nonzero_mask, zero_mask, _ = mask_tensor_with_masks(x, mask_zero_ratio, mask_nonzero_ratio)
            x, nonzero_mask, zero_mask, _ = mask_tensor_with_masks(latent_data, mask_zero_ratio, mask_nonzero_ratio)
            ob_cond = torch.tensor(x).float().to(device)
            nonzero_mask = torch.tensor(nonzero_mask).to(device)
            zero_mask = torch.tensor(zero_mask).to(device)
            x = torch.tensor(x).to(device)
            cond_emb = text_cond(cond, tokenizer, text_model).to(device)
            stage_emb = stage_cond(stage, tokenizer, text_model).to(device)
            #cond_emb = torch.randn(64, 768).to(device)
            #stage_emb = stage_cond(stage, tokenizer, text_model).to(device)
            noise = torch.randn(x.shape).to(device)
            timesteps = torch.randint(1, diffusion_step, (x.shape[0],)).long()
            timesteps = timesteps.to(device)
            x_t = noise_scheduler.add_noise(x.to(device),
                                            noise,
                                            timesteps=timesteps)

            # mask = torch.tensor(mask).to(device)
            # mask = (1-((torch.rand(x.shape[1]) < mask_ratio).int())).to(device)
            mask = np.zeros((x.shape[0], x.shape[1]))
            mask[x.cpu() != 0] = 1

            #x_noisy = x_t.to(device) * (1 - torch.tensor(mask).to(device)) + x * torch.tensor(mask).to(device)

            #x_noisy = x_t.to(device) * torch.tensor(nonzero_mask).to(device) + x * torch.tensor(1-nonzero_mask).to(device)
            x_noisy = x_t
            ## x_noisy = x_t * (1 - nonzero_mask) + x * nonzero_mask
            ###############x_noisy = x_t.to(device) * nonzero_mask + x * (1 - nonzero_mask)
            ######x_noisy = x_t.to(device) * (zero_mask + nonzero_mask) + x * (1 - (zero_mask + nonzero_mask))
            # x_noisy = x_t.to(device) * (nonzero_mask + zero_mask) + x * (1 - nonzero_mask)

            #noise_pred = model(x_noisy, t=timesteps.to(device), y=cond_emb, a=age_cond, ob=x_cond.to(device))  # 去掉了, z=celltype
            noise_pred = model(x_noisy, t=timesteps.to(device), y=cond_emb, a=age_cond, ob=ob_cond.to(device), stage=stage_emb.to(device))
            loss = criterion(noise * nonzero_mask, noise_pred * nonzero_mask)

            # loss = criterion(noise*nonzero_mask, noise_pred*nonzero_mask, noise*zero_mask, noise_pred*zero_mask)

            ### loss = criterion(noise * nonzero_mask, noise_pred * nonzero_mask)
            # loss = criterion(noise * nonzero_mask, noise_pred * nonzero_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss = epoch_loss / (i + 1)  # type: ignore

        current_lr = optimizer.param_groups[0]['lr']

        # 更新tqdm的描述信息
        if is_tqdm:
            t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.5f}, lr:{current_lr:.2e}')  # type: ignore

        if is_tune:
            session.report({'loss': epoch_loss})
