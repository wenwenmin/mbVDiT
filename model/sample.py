import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from DiTT.preprocess.utils import calculate_rmse, calculate_pcc
from DiTT.preprocess.utils import mask_tensor_with_masks
from DiTT.preprocess.utils import mask_tensor_with_masks, text_cond, calculate_cosine_similarity, calculate_mae, stage_cond
from transformers import BertTokenizer, BertModel, BertConfig
from DiTT.sample_vae import Vae
import pandas as pd


model_name = "bert-base-uncased"  # 预训练的 BERT 模型
tokenizer = BertTokenizer.from_pretrained(model_name)
text_model = BertModel.from_pretrained(model_name)



def model_sample_diff(model, device, dataloader, total_sample, time, is_condi, ob_cond, condi_flag, vae_model):
    noise = []
    i = 0
    for x, x_cond, cond, age_cond, _, _, _, stage in dataloader: # 计算整个shape得噪声 一次循环算batch大小  加上了celltype 去掉了, celltype
        # x_cond = x.float().to(device) # x.float().to(device)
        # x, nonzero_mask, zero_mask, org_data = mask_tensor_with_masks(x, 0.5, 0.5)
        # cond_emb = text_cond(cond, tokenizer, text_model).to(device)
        cond_emb = text_cond(cond, tokenizer, text_model).to(device)
        stage_emb = stage_cond(stage, tokenizer, text_model).to(device)
        #cond_emb = torch.randn(64, 768).to(device)
        #stage_emb = stage_cond(stage, tokenizer, text_model).to(device)
        age_cond = torch.squeeze(age_cond).to(device)
        t = torch.from_numpy(np.repeat(time, x_cond.shape[0])).long().to(device)
        # celltype = celltype.to(device)

        _, _, _, latent_data = vae_model(torch.tensor(x).float().to(device))
        if not is_condi:
            n = model(total_sample[i:i+len(x)], t, None) # 一次计算batch大小得噪声
        else:
            n = model(total_sample[i:i+len(x)], t, cond_emb, age_cond, ob=torch.tensor(latent_data).float().to(device), stage=stage_emb, condi_flag=condi_flag) # 加上了celltype 去掉了, celltype
        noise.append(n)
        i = i+len(x)
    noise = torch.cat(noise, dim=0)
    return noise

def sample_diff(model,
                dataloader,
                noise_scheduler,
                mask_nonzero_ratio = None,
                mask_zero_ratio = None,
                gt=None,
                device=torch.device('cuda:0'),
                num_step=1000,
                sample_shape=(7060, 2000),
                is_condi=False,
                sample_intermediate=200,
                org_data=None,
                model_pred_type: str = 'x_start',
                is_classifier_guidance=False,
                omega=0.1,
                nonzero_mask=None,
                zero_mask=None,
                is_tqdm = True):
    model.eval()
    vae_model = Vae().to(device)
    #vae_model.load_state_dict(torch.load('D:\\TCGA\mbVDiT_data\\WXS\\solid\\stad\\no_pretrain.pth'))
    vae_model.load_state_dict(torch.load('D:\\TCGA\mbVDiT_data\\WXS\\solid\\stad\\pretrain_tune.pth'))

    x_t = torch.randn(sample_shape[0], sample_shape[1]).to(device)
    timesteps = list(range(num_step))[::-1]  # 倒序
    # x_t, mask_nonzero, mask_zero = mask_tensor_with_masks(x_t, mask_zero_ratio, mask_nonzero_ratio)
    # mask = torch.tensor(mask_nonzero).to(device)

    #_, _, _, latent_data = vae_model(gt.to(device))
    # gt, nonzero_mask, zero_mask, org_data = mask_tensor_with_masks(gt, 0.1, 0.1)

    mask = np.zeros((sample_shape[0], sample_shape[1]))
    #mask[gt.cpu() != 0] = 1
    #mask[gt != 0] = 1

    #mask = nonzero_mask + zero_mask
    # mask = nonzero_mask
    #mask = torch.tensor(mask).to(device)
    #gt = torch.tensor(gt).to(device)
    ####x_t = x_t * mask + gt * (1 - mask)
    #x_t = x_t * (1 - mask) + gt * mask
    x_t = x_t
    ##x_t = x_t * (1 - mask) + gt
    # x_t = x_t + gt
    '''
    x_t = x_t * (1 - mask) + gt * mask
    '''
    # x_t = x_t * (1 - mask)
    if sample_intermediate:
        timesteps = timesteps[:sample_intermediate]

    ts = tqdm(timesteps)
    for t_idx, time in enumerate(ts):
        ts.set_description_str(desc=f'time: {time}')
        with torch.no_grad():
            # 输出噪声
            model_output = model_sample_diff(model,
                                        device=device,
                                        dataloader=dataloader,
                                        total_sample=x_t,  # x_t
                                        time=time,  # t
                                        is_condi=is_condi,
                                        ob_cond=None,
                                        vae_model=vae_model,
                                        condi_flag=True)
            if is_classifier_guidance:
                model_output_uncondi = model_sample_diff(model,
                                                    device=device,
                                                    dataloader=dataloader,
                                                    total_sample=x_t,
                                                    time=time,
                                                    is_condi=is_condi,
                                                    condi_flag=False)
                model_output = (1 + omega) * model_output - omega * model_output_uncondi

        # 计算x_{t-1}
        x_t, _ = noise_scheduler.step(model_output,  # 一般是噪声
                                         torch.from_numpy(np.array(time)).long().to(device),
                                         x_t,
                                         model_pred_type=model_pred_type)
        #epoch_pcc = calculate_pcc(x_t, gt, nonzero_mask.to(device), zero_mask.to(device))
        #epoch_rmse = calculate_rmse(x_t, gt, nonzero_mask.to(device), zero_mask.to(device))

        re_data = vae_model.decoder(x_t)

        epoch_pcc = calculate_pcc(re_data, torch.tensor(org_data).to(device), nonzero_mask, zero_mask)
        epoch_cos = calculate_cosine_similarity(re_data, torch.tensor(org_data).to(device), nonzero_mask, zero_mask)
        epoch_rmse = calculate_rmse(re_data, torch.tensor(org_data).to(device), nonzero_mask, zero_mask)
        epoch_mae = calculate_mae(re_data, torch.tensor(org_data).to(device), nonzero_mask, zero_mask)
        ts.set_postfix_str(f'PCC:{epoch_pcc:.5f}, COS:{epoch_cos:.5f}, RMSE:{epoch_rmse:.5f}, MAE:{epoch_mae:.5f}')
        if mask is not None:
            #x_t = x_t * (1. - mask) + mask * gt  # 真实值和预测部分的拼接
            x_t = x_t
            #x_t = x_t * (1 - mask) + gt
            #x_t = x_t + gt
            ### x_t = x_t * mask + gt * (1 - mask)

        if time == 0 and model_pred_type == 'x_start':
            # 如果直接预测 x_0 的话，最后一步直接输出
            sample = model_output


    # recon_x = x_t.detach().cpu().numpy()
    #pcc = calculate_pcc(x_t, gt, nonzero_mask.to(device), zero_mask.to(device))
    #rmse = calculate_rmse(x_t, gt, nonzero_mask.to(device), zero_mask.to(device))
    re_data = vae_model.decoder(x_t)
    pcc = calculate_pcc(re_data, torch.tensor(org_data).to(device), nonzero_mask, zero_mask)
    rmse = calculate_rmse(re_data, torch.tensor(org_data).to(device), nonzero_mask, zero_mask)
    cos = calculate_cosine_similarity(re_data, torch.tensor(org_data).to(device), nonzero_mask, zero_mask)
    mae = calculate_mae(re_data, torch.tensor(org_data).to(device), nonzero_mask, zero_mask)
    print(pcc, cos, rmse, mae)
    recon_x = re_data.detach().cpu().numpy()
    return recon_x

