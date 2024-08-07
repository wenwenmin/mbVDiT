import anndata as ad
import numpy as np
import pandas as pd
import sys
import pickle
import os
import datetime
import time as tm
from functools import partial
import scipy.stats as st
from scipy.stats import wasserstein_distance
import scipy.stats
import copy
from sklearn.model_selection import KFold
import pandas as pd
import multiprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import warnings
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import distance_matrix
from sklearn.metrics import matthews_corrcoef
from scipy import stats
import seaborn as sns
import torch
from scipy.spatial.distance import cdist
import h5py
import time
import sys
import tangram as tg
import pickle
import yaml
import argparse
from os.path import join
from IPython.display import display
from DiTT.model.diff_model import DiT_diff
from DiTT.model.diff_scheduler import NoiseScheduler
from DiTT.model.diff_train import normal_train_diff
from DiTT.model.sample import sample_diff
from DiTT.preprocess.result_analysis import clustering_metrics
from DiTT.preprocess.utils import *
from DiTT.preprocess.data import *
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--sc_data", type=str, default='_microdata.h5ad')
parser.add_argument("--st_data", type=str, default='_microdata.h5ad')
parser.add_argument("--document", type=str, default='data_stad')
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--batch_size", type=int, default=256)  # 256
parser.add_argument("--hidden_size", type=int, default=64)  # 64
parser.add_argument("--diffusion_step", type=int, default=500) #200
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--depth", type=int, default=16) # 16
parser.add_argument("--noise_std", type=float, default=10)
parser.add_argument("--head", type=int, default=16) # 16
parser.add_argument("--mask_nonzero_ratio", type=float, default=0.1)
parser.add_argument("--mask_zero_ratio", type=float, default=0)
parser.add_argument("--seed", type=int, default=random.randint(2000, 5000)) # random.randint(2000, 5000)
args = parser.parse_args()

print(os.getcwd())
print(torch.cuda.get_device_name(torch.cuda.current_device()))


def train_valid_test():
    seed_everything(args.seed)
    st_path = 'datasets/' + args.document + '/st/' + args.document + args.st_data
    sc_path = 'datasets/' + args.document + '/sc/' + args.document + args.sc_data

    text_path = "D:/ucec_sample_type.csv"

    directory = 'save/' + args.document + '_ckpt/' + args.document + '_scdiff'
    # currt_time = datetime.datetime.now().strftime("%Y%m%d")

    if not os.path.exists(directory):
        os.makedirs(directory)
    # save_path = os.path.join(directory, f'{currt_time}.pt')
    save_path = os.path.join(directory, args.document + '.pt')


    dataset = ConditionalDiffusionDataset(sc_path, st_path, text_path)
    train_dataset, valid_dataset, test_dataset = split_dataset(dataset,
                                                               train_ratio=1,
                                                               val_ratio=0,
                                                               test_ratio=0,
                                                               random_state=args.seed)

    # all_data_matrix = torch.stack([data for data, _, _, _ in valid_dataset])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    #valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    #test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    cell_num = dataset.sc_data.shape[0]
    spot_num = dataset.st_data.shape[0]
    sc_gene_num = dataset.sc_data.shape[1]
    st_gene_num = dataset.st_data.shape[1]
    # mask_1 = (1 - ((torch.rand(st_gene_num) < args.mask_ratio).int())).to(args.device)
    # mask_0 = (1 - ((torch.rand(st_gene_num) < args.mask_ratio).int())).to(args.device)
    age = pd.read_csv("D:/ucec_sample_age.csv", header=0, index_col=False)
    max_age = int(age.values.max())

    model = DiT_diff(
        st_input_size=32, #128
        condi_input_size=32,
        hidden_size=args.hidden_size,
        depth=args.depth,
        max_age=max_age,
        num_heads=args.head,
        classes=6,
        mlp_ratio=4.0,
        dit_type='cross_dit' #cross_dit
    )

    model.to(args.device)
    diffusion_step = args.diffusion_step

    model.train()

    if not os.path.isfile(save_path):
        normal_train_diff(model,
                          dataloader=train_dataloader,
                          lr=args.learning_rate,
                          num_epoch=args.epoch,
                          diffusion_step=diffusion_step,
                          device=args.device,
                          pred_type='noise',
                          mask_nonzero_ratio=args.mask_nonzero_ratio,
                          mask_zero_ratio=args.mask_zero_ratio)
        torch.save(model.state_dict(), save_path)
    else:
        model.load_state_dict(torch.load(save_path))

    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='quadratic'
    )

    model.eval()
    # valid_gt = torch.stack([data for data, _ in valid_dataset])
    # imputation = sample_diff(model,
    #                          device=args.device,
    #                          dataloader=valid_dataloader,
    #                          noise_scheduler=noise_scheduler,
    #                          mask=mask,
    #                          gt=valid_gt,
    #                          num_step=diffusion_step,
    #                          sample_shape=(valid_gt.shape[0], valid_gt.shape[1]),
    #                          is_condi=True,
    #                          sample_intermediate=diffusion_step,
    #                          model_pred_type='noise',
    #                          is_classifier_guidance=False,
    #                          omega=0.9
    #                          )

    with torch.no_grad():
       test_gt = torch.stack([data for data, _, _, _, _, _, _, _ in train_dataset])
       org_data_array = torch.stack([data for _, _, _, _, _, _, data, _ in train_dataset])
       nonzero_mask = torch.stack([data for _, _, _, _, data, _, _, _ in train_dataset])
       mask = np.array(nonzero_mask) == 1
       org_data = np.array(org_data_array.cpu().detach().numpy())[mask]
       # test_gt = torch.randn(len(test_dataset), 249)
       prediction = sample_diff(model,
                                device=args.device,
                                dataloader=train_dataloader,
                                noise_scheduler=noise_scheduler,
                                mask_nonzero_ratio=args.mask_nonzero_ratio,
                                mask_zero_ratio=args.mask_zero_ratio,
                                gt=test_gt,
                                num_step=diffusion_step,
                                #sample_shape=(test_gt.shape[0], test_gt.shape[1]),
                                sample_shape=(test_gt.shape[0], 32), #128
                                is_condi=True,
                                sample_intermediate=diffusion_step,
                                model_pred_type='noise',
                                is_classifier_guidance=False,
                                omega=0.9,
                                org_data=org_data,
                                nonzero_mask=nonzero_mask
                                )

    return prediction, test_gt+org_data_array



Data =  args.document
outdir = 'result/' + Data +'/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

hyper_directory = 'save/'+Data+'_ckpt/'+Data+'_hyper/'
hyper_file = Data + '_hyperameters.yaml'
hyper_full_path = os.path.join(hyper_directory, hyper_file)
if not os.path.exists(hyper_directory):
    os.makedirs(hyper_directory)
args_dict = vars(args)
with open(hyper_full_path, 'w') as yaml_file:
    yaml.dump(args_dict, yaml_file)

prediction_result, ground_truth = train_valid_test()
result_mask = ground_truth
result_mask = np.where(result_mask != 0, 1, 0)

prediction_result_final = ground_truth + prediction_result*(1-result_mask)

# st_common_gene = pd.read_csv("D:/OSC.csv", header=0).iloc[:, 1:].columns
# st_unique_gene = pd.read_csv("D:/OSC.csv", header=0).iloc[:, 1:].columns
# gene_name = st_common_gene + st_unique_gene
# pred_result = pd.DataFrame(prediction_result, columns=[gene_name])
# original = pd.DataFrame(ground_truth.numpy(), columns=[gene_name])
# pred_result.to_csv(outdir + '/prediction.csv', header=True, index=True)
# original.to_csv(outdir + '/original.csv', header=True, index=True)
'''
st_common_gene = pd.read_csv("D:/bic_names.csv", header=0).T
st_common_gene = st_common_gene.values[0]
'''
pred_result = pd.DataFrame(prediction_result_final)
original = pd.DataFrame(ground_truth.numpy())
pred_result.to_csv(outdir + '/prediction.csv', header=True, index=True)
original.to_csv(outdir + '/original.csv', header=True, index=True)
#
# pred_result = pd.DataFrame(pred, columns=[st_common_gene+st_unique_gene])
#
# print(pred_result)