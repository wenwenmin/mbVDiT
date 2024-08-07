import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
import scipy
import os
from DiT.preprocess.data import reindex
from DiT.preprocess.utils import calculate_zero_percentage, ensure_dir_exists
import argparse
parser = argparse.ArgumentParser(description='Process some datasets')
parser.add_argument('--sc_data', type=str, default='la_test.csv')
parser.add_argument('--st_data', type=str, default='la_test.csv')
parser.add_argument('--document', type=str, default='data_hv')
args = parser.parse_args()
print(os.getcwd())

sc_path = '../datasets/' + args.document + '/' + args.sc_data
st_path = '../datasets/' + args.document + '/' + args.st_data
# adata_seq = sc.read(sc_path, sep='\t', first_column_names=True).T
adata_seq = sc.read(sc_path, sep='\t', first_column_names=True)
adata_spatial = sc.read(st_path, sep='\t')
print('sc dataset shape:', adata_seq.shape)
print('st dataset shape:', adata_spatial.shape)
adata_seq_copy = adata_seq.copy()
adata_spatial_copy = adata_spatial.copy()

sc.pp.normalize_total(adata_spatial_copy, target_sum=1e5) #1e4
sc.pp.log1p(adata_spatial_copy)

sc.pp.normalize_total(adata_seq_copy, target_sum=1e5)
sc.pp.log1p(adata_seq_copy)

'''
sc.pp.filter_genes(adata_seq_copy, min_cells=500)
sc.pp.highly_variable_genes(adata_seq_copy, n_top_genes=600)
adata_seq_copy = adata_seq_copy[:, adata_seq_copy.var.highly_variable]
'''


# adata = ad.AnnData(adata_spatial_copy)
# adata.write('../datasets/' + args.document + '/' + '_OSC_test.h5ad')
adata_seq_copy.write('../datasets/' + args.document + '/' + '_HV_1000.h5ad')
'''
f = adata_seq_copy.var_names
f = pd.DataFrame(f)
f.to_csv('D:/all_names.csv', index=False)
'''
print()