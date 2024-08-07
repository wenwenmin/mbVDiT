import pandas as pd
import numpy as np
import scanpy as sc
import scipy.stats as st
import os
import seaborn as sns
import scipy
from scipy.sparse import csr_matrix
import torch
import random
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, homogeneity_score, \
    normalized_mutual_info_score
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn

def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())

def mask_tensor_with_masks(X, mask_zero_ratio, mask_nonzero_ratio, decice = 'cuda:0'):

    nonzero_mask = np.zeros((X.shape[0], X.shape[1]))
    zero_mask = np.zeros((X.shape[0], X.shape[1]))

    nonzero_indices = np.transpose(np.nonzero(X.cpu().detach().numpy()))
    #num_indices_to_choose = int(mask_nonzero_ratio * len(nonzero_indices[0]))
    num_indices_to_choose = int(mask_nonzero_ratio * len(nonzero_indices))
    # chosen_indices = np.random.choice(num_indices_to_choose, size=num_indices_to_choose, replace=False)
    #chosen_indices_row = np.random.choice(len(nonzero_indices[0]), size=num_indices_to_choose, replace=False)
    chosen_indices_row = np.random.choice(len(nonzero_indices), size=num_indices_to_choose, replace=False)
    #chosen_indices_col = np.random.choice(nonzero_indices[1], size=num_indices_to_choose, replace=False)


    #for idx in chosen_indices_row:
    #    row, col = nonzero_indices[0][idx], nonzero_indices[1][idx]
     #   nonzero_mask[row, col] = 1

    for idx in chosen_indices_row:
        row, col = nonzero_indices[idx][0], nonzero_indices[idx][1]
        nonzero_mask[row, col] = 1

    zero_indices = np.where(X.cpu().detach().numpy() == 0)
    #num_indices_to_choose = int(mask_zero_ratio * len(zero_indices[0]))
    # chosen_indices = np.random.choice(num_indices_to_choose, size=num_indices_to_choose, replace=False)

    #chosen_indices_row = np.random.choice(len(zero_indices[0]), size=num_indices_to_choose, replace=False)
    #chosen_indices_col = np.random.choice(zero_indices[1], size=num_indices_to_choose, replace=False)

    #for idx in chosen_indices_row:
     #   row, col = zero_indices[0][idx], zero_indices[1][idx]
      #  zero_mask[row, col] = 1

    num_indices_to_choose = int(mask_zero_ratio * len(zero_indices))
    # chosen_indices = np.random.choice(num_indices_to_choose, size=num_indices_to_choose, replace=False)

    chosen_indices_row = np.random.choice(len(zero_indices), size=num_indices_to_choose, replace=False)
    # chosen_indices_col = np.random.choice(zero_indices[1], size=num_indices_to_choose, replace=False)

    for idx in chosen_indices_row:
        row, col = zero_indices[idx][0], zero_indices[idx][1]
        zero_mask[row, col] = 1

    print()
    # mask = np.bitwise_or(nonzero_mask.astype(int), zero_mask.astype(int))
    # mask_X = X.cpu() * (1 - nonzero_mask)



    #all_mask = np.array(nonzero_mask + zero_mask)
    all_mask = np.array(nonzero_mask)
    mask = all_mask == 1

    # org_data = np.array(X.cpu().detach().numpy())[mask]
    org_data = X.cpu().detach().numpy() * nonzero_mask

    mask_X = X.cpu().detach().numpy() * (1 - nonzero_mask)

    return mask_X, nonzero_mask, zero_mask, org_data

def text_cond(text_index, tokenizer, text_model):

    text_data = []
    for i in text_index:
        if i == float(1):
            text_data.append('Primary Tumor')
        if i == float(2):
            text_data.append('Recurrent Tumor')
        if i == float(3):
            text_data.append('Blood Derived Normal')
        if i == float(4):
            text_data.append('Solid Tissue Normal')
        if i == float(5):
            text_data.append('Metastatic')
        if i == float(6):
            text_data.append('Additional - New Primary')
        if i == float(7):
            text_data.append('Primary Blood Derived Cancer - Peripheral Blood')
        if i == float(8):
            text_data.append('Additional Metastatic')

    text_condi = text_data
    encoded_input = tokenizer(text_condi, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = text_model(**encoded_input)
    text_embeddings = outputs.pooler_output
    # linear_layer = nn.Linear(768, 650)
    # embeddings_650d = linear_layer(text_embeddings)
    return text_embeddings

def stage_cond(text_index, tokenizer, text_model):

    text_data = []
    for i in text_index:
        if i == float(1):
            text_data.append('Not available')
        if i == float(2):
            text_data.append('Stage I')
        if i == float(3):
            text_data.append('Stage IA')
        if i == float(4):
            text_data.append('Stage IIA')
        if i == float(5):
            text_data.append('Stage IIB')
        if i == float(6):
            text_data.append('Stage III')
        if i == float(7):
            text_data.append('Stage IIIA')
        if i == float(8):
            text_data.append('Stage IIIB')
        if i == float(9):
            text_data.append('Stage IIIC')
        if i == float(10):
            text_data.append('Stage IV')
        if i == float(11):
            text_data.append('Stage Tis')
        if i == float(12):
            text_data.append('Stage X')
        if i == float(13):
            text_data.append('Stage IB')
        if i == float(14):
            text_data.append('Stage II')
        if i == float(15):
            text_data.append('Stage IIC')
        if i == float(16):
            text_data.append('Stage IVA')
        if i == float(17):
            text_data.append('Stage IVB')
        if i == float(18):
            text_data.append('Stage IVC')

    text_condi = text_data
    encoded_input = tokenizer(text_condi, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = text_model(**encoded_input)
    text_embeddings = outputs.pooler_output
    # linear_layer = nn.Linear(768, 650)
    # embeddings_650d = linear_layer(text_embeddings)
    return text_embeddings


def clustering_metrics(adata, target, pred, mode="AMI"):
    """
    Evaluate clustering performance.
   
    Parameters
    ----------
    adata
        AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    target
        Key in `adata.obs` where ground-truth spatial domain labels are stored.
    pred
        Key in `adata.obs` where clustering assignments are stored.
       
    Returns
    -------
    ami
        Adjusted mutual information score.
    ari
        Adjusted Rand index score.
    homo
        Homogeneity score.
    nmi
        Normalized mutual information score.

    """
    if(mode=="AMI"):
        ami = adjusted_mutual_info_score(adata.obs[target], adata.obs[pred])
        print("AMI ",ami)
        return ami
    elif(mode=="ARI"):
        ari = adjusted_rand_score(adata.obs[target], adata.obs[pred])
        print("ARI ",ari)
        return ari
    elif(mode=="Homo"):
        homo = homogeneity_score(adata.obs[target], adata.obs[pred])
        print("Homo ",homo)
        return homo
    elif(mode=="NMI"):
        nmi = normalized_mutual_info_score(adata.obs[target], adata.obs[pred])
        print("NMI ", nmi)
        return nmi

from scipy.stats import pearsonr
'''
def calculate_pcc(tensor1, tensor2, nonzero_mask, zero_mask):
    pccs = []
    for i in range(tensor1.shape[1]):
        column1 = tensor1.cpu()[:, i]  # 获取第一个矩阵的第i列
        column2 = tensor2.cpu()[:, i]
        pccs.append(pearsonr(column1, column2)[0])
    f = np.array(pccs).mean()
    return f
'''

def calculate_pcc(tensor1, tensor2, nonzero_mask, zero_mask):
    #all_mask = np.array(nonzero_mask + zero_mask)
    all_mask = np.array(nonzero_mask)
    mask = all_mask == 1
    t1 = np.array(tensor1.cpu())
    t1 = t1[mask]

    t2 = np.array(tensor2.cpu())
    f = pearsonr(t1, t2)[0]
    return f

'''
def calculate_rmse(tensor1, tensor2):
    squared_diff = (tensor1 - tensor2) ** 2
    mean_squared_diff = squared_diff.mean()
    rmse = torch.sqrt(mean_squared_diff)
    return rmse
'''
def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())

def calculate_rmse(tensor1, tensor2, nonzero_mask, zero_mask):
    #all_mask = np.array(nonzero_mask + zero_mask)
    all_mask = np.array(nonzero_mask)
    mask = all_mask == 1
    t1 = np.array(tensor1.cpu())
    t1 = t1[mask]

    t2 = np.array(tensor2.cpu())

    '''
    t1 = pd.DataFrame(t1)
    t1 = normalize_data(t1)
    t1 = np.array(t1)

    t2 = pd.DataFrame(t2)
    t2 = normalize_data(t2)
    t2 = np.array(t2)
    '''
    # 计算差值
    diff = t1 - t2

    # 计算平方
    squared_diff = diff ** 2

    # 计算平方差的平均值
    mean_squared_diff = np.mean(squared_diff)

    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mean_squared_diff)

    return rmse

def calculate_cosine_similarity(tensor1, tensor2, nonzero_mask, zero_mask):
    #all_mask = np.array(nonzero_mask + zero_mask)
    all_mask = np.array(nonzero_mask)
    mask = all_mask == 1
    t1 = np.array(tensor1.cpu())
    a = t1[mask]

    b = np.array(tensor2.cpu())

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)

    return similarity

def calculate_mae(tensor1, tensor2, nonzero_mask, zero_mask):
    #all_mask = np.array(nonzero_mask + zero_mask)
    all_mask = np.array(nonzero_mask)
    mask = all_mask == 1
    t1 = np.array(tensor1.cpu())
    a = t1[mask]

    b = np.array(tensor2.cpu())

    '''
    t1 = pd.DataFrame(a)
    t1 = normalize_data(t1)
    t1 = np.array(t1)

    t2 = pd.DataFrame(b)
    t2 = normalize_data(t2)
    t2 = np.array(t2)
    '''

    mae = np.mean(np.abs(a - b))
    return mae

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42):
    # assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1."

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_zero_percentage(adata):
    data_matrix = adata.X
    total_elements = data_matrix.shape[0] * data_matrix.shape[1]

    if scipy.sparse.issparse(data_matrix):
        nonzero_elements = data_matrix.count_nonzero()
        zero_elements = total_elements - nonzero_elements
    else:
        zero_elements = np.count_nonzero(data_matrix == 0)

    zero_percentage = (zero_elements / total_elements) * 100
    return zero_percentage

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def add_constant_to_sparse_matrix(adata):
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray()  # Convert to dense
    adata.X += 1e-10  # Add small constant
    adata.X = csr_matrix(adata.X)  # Convert back to sparse

    return adata


def cal_ssim(im1, im2, M):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12

    return ssim


def scale_max(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.max()
        result = pd.concat([result, content], axis=1)
    return result


def scale_z_score(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = st.zscore(content)
        content = pd.DataFrame(content, columns=[label])
        result = pd.concat([result, content], axis=1)
    return result


def scale_plus(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.sum()
        result = pd.concat([result, content], axis=1)
    return result


def logNorm(df):
    df = np.log1p(df)
    df = st.zscore(df)
    return df


class CalculateMeteics:
    def __init__(self, raw_data, adata_data, genes_name, impute_count_file, prefix, metric):
        self.impute_count_file = impute_count_file
        self.raw_count = pd.DataFrame(raw_data, columns=genes_name)
        self.raw_count.columns = [x.upper() for x in self.raw_count.columns]
        self.raw_count = self.raw_count.T
        self.raw_count = self.raw_count.loc[~self.raw_count.index.duplicated(keep='first')].T
        self.raw_count = self.raw_count.fillna(1e-20)

        self.adata_data = adata_data

        self.impute_count = pd.read_csv(impute_count_file, header=0, index_col=0)
        self.impute_count.columns = [x.upper() for x in self.impute_count.columns]
        self.impute_count = self.impute_count.T
        self.impute_count = self.impute_count.loc[~self.impute_count.index.duplicated(keep='first')].T
        self.impute_count = self.impute_count.fillna(1e-20)
        self.prefix = prefix
        self.metric = metric

    def SSIM(self, raw, impute, scale='scale_max'):
        print('---------Calculating SSIM---------')
        if scale == 'scale_max':
            raw = scale_max(raw)
            impute = scale_max(impute)
        else:
            print('Please note you do not scale data by scale max')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    ssim = 0
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    M = [raw_col.max(), impute_col.max()][raw_col.max() > impute_col.max()]
                    raw_col_2 = np.array(raw_col)
                    raw_col_2 = raw_col_2.reshape(raw_col_2.shape[0], 1)
                    impute_col_2 = np.array(impute_col)
                    impute_col_2 = impute_col_2.reshape(impute_col_2.shape[0], 1)
                    ssim = cal_ssim(raw_col_2, impute_col_2, M)

                ssim_df = pd.DataFrame(ssim, index=["SSIM"], columns=[label])
                result = pd.concat([result, ssim_df], axis=1)
        else:
            print("columns error")
            return pd.DataFrame()

        print(result)
        return result

    def PCC(self, raw, impute, scale = 'scale_z_score'):
        print('---------Calculating PCC---------')
        if scale == 'scale_z_score':
            raw = scale_z_score(raw)
            impute = scale_z_score(impute)
        else:
            print('Please note you do not scale data by logNorm')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    pearsonr = 0
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    impute_col = impute_col.fillna(1e2)
                    raw_col = raw_col.fillna(1e2)
                    pearsonr, _ = st.pearsonr(raw_col, impute_col)
                pcc_df = pd.DataFrame(pearsonr, index=["PCC"], columns=[label])
                result = pd.concat([result, pcc_df], axis=1)
        else:
            print("columns error")

        print(result)
        return result

    def JS(self, raw, impute, scale='scale_plus'):
        print('---------Calculating JS---------')
        if scale == 'scale_plus':
            raw = scale_plus(raw)
            impute = scale_plus(impute)
        else:
            print('Please note you do not scale data by plus')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    JS = 1
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    raw_col = raw_col.fillna(1e-20)
                    impute_col = impute_col.fillna(1e-20)
                    M = (raw_col + impute_col) / 2
                    JS = 0.5 * st.entropy(raw_col, M) + 0.5 * st.entropy(impute_col, M)
                JS_df = pd.DataFrame(JS, index=["JS"], columns=[label])
                result = pd.concat([result, JS_df], axis=1)
        else:
            print("columns error")

        print(result)
        return result

    def RMSE(self, raw, impute, scale='zscore'):
        print('---------Calculating RMSE---------')
        if scale == 'zscore':
            raw = scale_z_score(raw)
            impute = scale_z_score(impute)
        else:
            print('Please note you do not scale data by zscore')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    RMSE = 1.5
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    RMSE = np.sqrt(((raw_col - impute_col) ** 2).mean())

                RMSE_df = pd.DataFrame(RMSE, index=["RMSE"], columns=[label])
                result = pd.concat([result, RMSE_df], axis=1)
        else:
            print("columns error")

        print(result)
        return result

    def cluster(self, adata_data, impu, scale=None):
        print('---------Calculating cluster---------')

        cpy_x = adata_data.copy()
        cpy_x.X = impu

        sc.tl.pca(adata_data)
        sc.pp.neighbors(adata_data, n_pcs=30, n_neighbors=30)
        sc.tl.leiden(adata_data)
        tmp_adata1 = adata_data

        sc.tl.pca(cpy_x)
        sc.pp.neighbors(cpy_x, n_pcs=30, n_neighbors=30)
        sc.tl.leiden(cpy_x)
        tmp_adata2 = cpy_x

        tmp_adata2.obs['class'] = tmp_adata1.obs['leiden']

        # tmp_adata2 = get_N_clusters(cpy_x, 23, 'leiden') # merfish-mop 23类别
        # tmp_adata2.obs['class'] = adata_spatial2.obs['subclass_label']

        ari = clustering_metrics(tmp_adata2, 'class', 'leiden', "ARI")
        ami = clustering_metrics(tmp_adata2, 'class', 'leiden', "AMI")
        homo = clustering_metrics(tmp_adata2, 'class', 'leiden', "Homo")
        nmi = clustering_metrics(tmp_adata2, 'class', 'leiden', "NMI")
        result = pd.DataFrame([[ari, ami, homo, nmi]], columns=["ARI", "AMI", "Homo", "NMI"])
        return result

    def compute_all(self):
        raw = self.raw_count
        impute = self.impute_count
        prefix = self.prefix
        adata_data = self.adata_data
        SSIM_gene = self.SSIM(raw, impute)
        # Spearman_gene = self.SPCC(raw, impute)
        PCC_gene = self.PCC(raw, impute)
        JS_gene = self.JS(raw, impute)
        RMSE_gene = self.RMSE(raw, impute)

        cluster_result = self.cluster(adata_data, impute)

        result_gene = pd.concat([PCC_gene, SSIM_gene, RMSE_gene, JS_gene], axis=0)
        result_gene.T.to_csv(prefix + "_gene_Metrics.txt", sep='\t', header=1, index=1)

        cluster_result.to_csv(prefix + "_cluster_Metrics.txt", sep='\t', header=1, index=1)

        return result_gene


def CalDataMetric(Data, PATH, sp_data, sp_genes, adata_data, out_dir):
    print('We are calculating the : ' + Data + '\n')
    metrics = ['PCC(gene)',  'SSIM(gene)', 'RMSE(gene)', 'JS(gene)']
    metric = ['PCC', 'SSIM', 'RMSE', 'JS']
    impute_count_dir = PATH + Data
    impute_count = os.listdir(impute_count_dir)
    impute_count = [x for x in impute_count if x[-3:] == 'csv' and x != 'final_result.csv']
    methods = []
    if len(impute_count) != 0:
        medians = pd.DataFrame()
        for impute_count_file in impute_count:
            print(impute_count_file)
            if 'result_Tangram.csv' in impute_count_file:
                os.system('mv ' + impute_count_dir + '/result_Tangram.csv ' + impute_count_dir + '/Tangram_impute.csv')
            prefix = impute_count_file.split('_')[0]
            methods.append(prefix)
            prefix = impute_count_dir + '/' + prefix
            impute_count_file = impute_count_dir + '/' + impute_count_file
            # if not os.path.isfile(prefix + '_Metrics.txt'):
            print(impute_count_file)
            CM = CalculateMeteics(sp_data, adata_data, sp_genes, impute_count_file=impute_count_file, prefix=prefix,
                                  metric=metric)
            CM.compute_all()

            # 计算中位数
            median = []
            for j in ['_gene']:
                # j = '_gene'
                #     median = []
                tmp = pd.read_csv(prefix + j + '_Metrics.txt', sep='\t', index_col=0)
                for m in metric:
                    median.append(np.mean(tmp[m]))
                    # median.append((np.max(tmp[m]) + np.min(tmp[m]))/2)
            median = pd.DataFrame([median], columns=metrics)
            # 聚类指标

            clu = pd.read_csv(prefix + '_cluster' + '_Metrics.txt', sep='\t', index_col=0)
            median = pd.concat([median, clu], axis=1)
            medians = pd.concat([medians, median], axis=0)


        metrics += ["ARI", "AMI", "Homo", "NMI"]
        medians.columns = metrics
        medians.index = methods
        medians.to_csv(out_dir + '/final_result.csv', header=1, index=1)