import scipy
import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from scipy.sparse import issparse, csr
from anndata import AnnData
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from DiTT.preprocess.utils import mask_tensor_with_masks, text_cond
CHUNK_SIZE = 20000

def transform(sc, st):
    ''
class ConditionalDiffusionDataset(Dataset):
    def __init__(self, sc_path, st_path, text_path, transform=None):
        # 读取数据
        self.sc_data = sc.read_h5ad(sc_path)
        self.st_data = sc.read_h5ad(st_path)
        sc.pp.normalize_total(self.sc_data, target_sum=1e2)  # 1e4
        sc.pp.log1p(self.sc_data)
        sc.pp.normalize_total(self.st_data, target_sum=1e2)  # 1e4
        sc.pp.log1p(self.st_data)
        self.text_data = pd.read_csv("D:\\TCGA\mbVDiT_data\\WXS\\solid\\stad\\type.csv", header=0, index_col=False)

        self.age_data = pd.read_csv("D:/ucec_sample_age.csv", header=0, index_col=False)
        self.label_data = pd.read_csv("D:/osc_sample_label.csv", header=0, index_col=False)

        self.stage_data = pd.read_csv("D:\\TCGA\mbVDiT_data\\WXS\\solid\\stad\\stage.csv", header=0, index_col=False)
        self.transform = transform
        # 确定最大长度以便重复采样
        self.max_len = max(len(self.sc_data), len(self.st_data))
        sc_df = self.sc_data.to_df()
        st_df = self.st_data.to_df()

        # 重复采样较小的数据集
        if len(sc_df) < self.max_len:
            sc_df = sc_df.sample(n=self.max_len, replace=True)
        if len(st_df) < self.max_len:
            st_df = st_df.sample(n=self.max_len, replace=True)

        # 根据最大长度调整数据集大小
        sc_df = sc_df.iloc[:self.max_len]
        st_df = st_df.iloc[:self.max_len]

        # 将重采样后的DataFrame重新赋值回AnnData对象（如果需要）
        # 这部分取决于后续处理是否需要AnnData对象的特性
        # 否则，你可以直接使用DataFrame进行处理

        st_df, nonzero_mask, zero_mask, org_data = mask_tensor_with_masks(torch.tensor(st_df.values), 0, 0.1)

        self.sc_data = pd.DataFrame(sc_df)
        self.st_data = pd.DataFrame(st_df)
        self.nonzero_mask = nonzero_mask
        self.zero_mask = zero_mask
        self.org_data = org_data



    def __len__(self):
        return self.max_len

    def __getitem__(self, idx):
        sc_sample = self.sc_data.iloc[idx]
        st_sample = self.st_data.iloc[idx]
        if self.transform:
            sc_sample, st_sample = self.transform(sc_sample, st_sample)
        sc_sample = torch.tensor(sc_sample.values, dtype=torch.float32)
        st_sample = torch.tensor(st_sample.values, dtype=torch.float32)


        text_data = np.array(self.text_data['0'])
        text_data = text_data.tolist()
        text_index = []
        for i in text_data:
            if i == 'Primary Tumor':
                text_index.append(1)
            if i == 'Recurrent Tumor':
                text_index.append(2)
            if i == 'Blood Derived Normal':
                text_index.append(3)
            if i == 'Solid Tissue Normal':
                text_index.append(4)
            if i == 'Metastatic':
                text_index.append(5)
            if i == 'Additional - New Primary':
                text_index.append(6)
            if i == 'Primary Blood Derived Cancer - Peripheral Blood':
                text_index.append(7)
            if i == 'Additional Metastatic':
                text_index.append(8)
        text_data = torch.tensor(text_index, dtype=torch.float32)

        stage_data = np.array(self.stage_data['0'])
        stage_data = stage_data.tolist()
        stage_index = []
        for i in stage_data:
            if type(i) == float:
                stage_index.append(1)
            if i == 'Stage I':
                stage_index.append(2)
            if i == 'Stage IA':
                stage_index.append(3)
            if i == 'Stage IIA':
                stage_index.append(4)
            if i == 'Stage IIB':
                stage_index.append(5)
            if i == 'Stage III':
                stage_index.append(6)
            if i == 'Stage IIIA':
                stage_index.append(7)
            if i == 'Stage IIIB':
                stage_index.append(8)
            if i == 'Stage IIIC':
                stage_index.append(9)
            if i == 'Stage IV':
                stage_index.append(10)
            if i == 'Stage Tis':
                stage_index.append(11)
            if i == 'Stage X':
                stage_index.append(12)
            if i == 'Stage IB':
                stage_index.append(13)
            if i == 'Stage II':
                stage_index.append(14)
            if i == 'Stage IIC':
                stage_index.append(15)
            if i == 'Stage IVA':
                stage_index.append(16)
            if i == 'Stage IVB':
                stage_index.append(17)
            if i == 'Stage IVC':
                stage_index.append(18)
        stage_data = torch.tensor(stage_index, dtype=torch.float32)

        age_data = torch.tensor(self.age_data.values, dtype=torch.long)[idx]
        #label_data = torch.tensor(self.label_data.values, dtype=torch.long)[idx]

        nonzero_mask = torch.tensor(self.nonzero_mask)[idx]
        zero_mask = torch.tensor(self.zero_mask)[idx]
        org_data = torch.tensor(self.org_data)[idx]

        #return st_sample, sc_sample, text_data[idx], age_data, torch.tensor(self.nonzero_mask)[idx], torch.tensor(self.zero_mask)[idx]
        return st_sample, sc_sample, text_data[idx], age_data, nonzero_mask, zero_mask, org_data, stage_data[idx]


def reindex(adata, genes, chunk_size=CHUNK_SIZE):
    """
    Reindex AnnData with gene list

    Parameters
    ----------
    adata
        AnnData
    genes
        gene list for indexing
    chunk_size
        chunk large data into small chunks

    Return
    ------
    AnnData
    """
    idx = [i for i, g in enumerate(genes) if g in adata.var_names]
    print('There are {} gene in selected genes'.format(len(idx)))
    if len(idx) == len(genes):
        adata = adata[:, genes]
    else:
        new_X = scipy.sparse.lil_matrix((adata.shape[0], len(genes)))
        for i in range(new_X.shape[0] // chunk_size + 1):
            new_X[i * chunk_size:(i + 1) * chunk_size, idx] = adata[i * chunk_size:(i + 1) * chunk_size, genes[idx]].X
        adata = AnnData(new_X.tocsr(), obs=adata.obs, var={'var_names': genes})
    return adata


def plot_hvg_umap(hvg_adata, color=['celltype'], save_filename=None):
    sc.set_figure_params(dpi=80, figsize=(3, 3))  # type: ignore
    hvg_adata = hvg_adata.copy()
    if save_filename:
        sc.settings.figdir = save_filename
        save = '.pdf'
    else:
        save = None
    # ideal gas equation

    sc.pp.scale(hvg_adata, max_value=10)
    sc.tl.pca(hvg_adata)
    sc.pp.neighbors(hvg_adata, n_pcs=30, n_neighbors=30)
    sc.tl.umap(hvg_adata, min_dist=0.1)
    sc.pl.umap(hvg_adata, color=color, legend_fontsize=10, ncols=2, show=None, save=save, wspace=1)
    return hvg_adata


def get_data_loader(data_ary: np.ndarray,
                    cell_type: np.ndarray,
                    batch_size: int = 512,
                    is_shuffle: bool = True,
                    ):
    data_tensor = torch.from_numpy(data_ary.astype(np.float32))
    cell_type_tensor = torch.from_numpy(cell_type.astype(np.float32))
    dataset = TensorDataset(data_tensor, cell_type_tensor)
    generator = torch.Generator(device='cuda')
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=is_shuffle, drop_last=False,
        generator=generator)  # , generator=torch.Generator(device = 'cuda')


def scale(adata):
    scaler = MaxAbsScaler()
    # 对adata.X按行进行归一化
    normalized_data = scaler.fit_transform(adata.X.T).T

    # 更新归一化后的数据到adata.X
    adata.X = normalized_data
    return adata


def data_augment(adata: AnnData, fixed: bool, noise_std):
    # 定义增强参数，例如噪声的标准差
    noise_stddev = noise_std
    augmented_adata = adata.copy()
    gene_expression = adata.X

    if fixed:
        augmented_adata.X = augmented_adata.X + np.full(gene_expression.shape, noise_stddev)
    else:
        # 对每个基因的表达值引入随机噪声
        augmented_adata.X = augmented_adata.X + np.abs(np.random.normal(0, noise_stddev, gene_expression.shape))

    merge_adata = adata.concatenate(augmented_adata, join='outer')

    return merge_adata




