import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import os

np.random.seed(3407)

blue_gold_cmap = LinearSegmentedColormap.from_list(
    'custom_blue_gold',
    ['#023858', '#045a8d', '#0570b0', '#3690c0', '#74a9cf', '#a6bddb', '#d0d1e6', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506']
)

'''
def plot_heatmap(expression_gt, matched_spot_expression_pred, top_k=500):
    # take mean of expression
    mean = np.mean(expression_gt, axis=1)
    # take ind of top 100
    ind = np.argpartition(mean, -top_k)[-top_k:]

    # Compute the correlation matrix
    corr_matrix = np.corrcoef(expression_gt[ind, :])
    dendrogram = sch.dendrogram(sch.linkage(corr_matrix, method='ward'), no_plot=True)
    cluster_idx = dendrogram['leaves']
    with open('cluster_idx_list_10x.txt', 'w') as f:
        for item in cluster_idx:
            f.write(f"{item}\n")  # 将每个元素写入文件，每个元素占一行
    cluster_idx = []

    with open('cluster_idx_list_10x.txt', 'r') as f:
        # 逐行读取文件内容，并添加到列表中
        for line in f:
            # 去除每行末尾的换行符，并转换为相应的数据类型（例如整数）
            cluster_idx.append(int(line.strip()))

    corr_matrix = np.corrcoef(matched_spot_expression_pred[ind, :].T)
    #corr_matrix = np.corrcoef(expression_gt[ind, :])
    corr_matrix_a = np.where(corr_matrix > 0.99, corr_matrix, corr_matrix - 0)
    corr_matrix = corr_matrix_a[cluster_idx, :]
    corr_matrix = corr_matrix[:, cluster_idx]

    # Reorder the correlation matrix and plot the heatmap
    plt.figure(dpi=300, figsize=(5, 5))
    sns.heatmap(corr_matrix, cmap=blue_gold_cmap, xticklabels=False, yticklabels=False, cbar=True, vmin=-1, vmax=1)
    #plt.colorbar()
    #plt.savefig(rf"C:\DELL\Desktop\mclSTExp_attention\suppleme_Fig\heatmap/mclSTExp_10x.pdf", format="pdf", bbox_inches="tight")
    plt.show()

VDiT_csv1 = pd.read_csv('D:\\PyCharm\\Py_Projects\\DiTT\\result\\data_ucec\\original.csv', index_col=0)
VDiT_csv2 = pd.read_csv('D:\\PyCharm\\Py_Projects\\DiTT\\result\\data_ucec\\prediction.csv', index_col=0)
DeepImpute_csv2 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/DeepImpute/DeepImpute_ucec_prediction.csv', index_col=0)
AutoImpute_csv2 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/AutoImpute/AutoImpute_ucec_prediction.csv', index_col=0)
DCA_csv2 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/DCA/DCA_ucec_prediction.csv', index_col=0)
CpG_csv2 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/CpG/CpG_ucec_prediction.csv', index_col=0)
scVI_csv2 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/scVI/scVI_ucec_prediction.csv', index_col=0)
DeepMicroGen_csv2 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/DeepMicroGen/DeepMicroGen_ucec_prediction.csv', index_col=0)

data_list = [VDiT_csv1, VDiT_csv2, DeepImpute_csv2, AutoImpute_csv2, DCA_csv2, CpG_csv2, scVI_csv2, DeepMicroGen_csv2]

for data in data_list:
    plot_heatmap(VDiT_csv1.values, data.values)
'''

'''
def map(dataset1, dataset2, i):

    scaler = MinMaxScaler()
    #scaler = StandardScaler()

    #dataset1 = scaler.fit_transform(dataset1)
    #dataset2 = scaler.fit_transform(dataset2)

    datasets = np.corrcoef(dataset1.T)
    map_ndarray = np.corrcoef(dataset2.T)
    datasets = scaler.fit_transform(datasets)
    map_ndarray = scaler.fit_transform(map_ndarray)
    #map_ndarray[np.isnan(map_ndarray)] = 0
    #datasets[np.isnan(datasets)] = 0

    yellow_cmap = LinearSegmentedColormap.from_list('custom_yellow', ['lightyellow', 'gold', 'orange'])

    blue_gold_cmap = LinearSegmentedColormap.from_list(
        'custom_blue_gold',
        ['#023858', '#045a8d', '#0570b0', '#3690c0', '#74a9cf', '#a6bddb', '#d0d1e6', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506']
    )
    dynamic_cmap = LinearSegmentedColormap.from_list(
        'custom_dynamic',
        ['#4B0082', '#9400D3', '#FF0000', '#FF4500', '#FFA500', '#FFFF00']
    )

    row_clusters = linkage(pdist(datasets, metric='euclidean'), method='ward') #dataset2  dataset1
    col_clusters = linkage(pdist(datasets.T, metric='euclidean'), method='ward')
    row_order = dendrogram(row_clusters, no_plot=True)['leaves']
    col_order = dendrogram(col_clusters, no_plot=True)['leaves']
    test_data = pd.DataFrame(map_ndarray)
    sorted_data = test_data.iloc[row_order, col_order]



    data_normalized = scaler.fit_transform(sorted_data)
    ax = sns.heatmap(sorted_data, cmap=blue_gold_cmap, cbar=True, annot=False)  #画图1用data_normalized   画图4用sorted_data

    save_dir = 'D:\\PyCharm\\Py_Projects\\DiTT\\heatmap_4\\KRCCC\\'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'heatmap_{:02d}.pdf'.format(i))
    #plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


VDiT_csv1 = pd.read_csv('D:\\PyCharm\\Py_Projects\\DiTT\\result\\data_hnsc\\original.csv', index_col=0)
VDiT_csv2 = pd.read_csv('D:\\PyCharm\\Py_Projects\\DiTT\\result\\data_hnsc\\prediction.csv', index_col=0)
DeepImpute_csv2 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/DeepImpute/DeepImpute_hnsc_prediction.csv', index_col=0)
AutoImpute_csv2 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/AutoImpute/AutoImpute_hnsc_prediction.csv', index_col=0)
DCA_csv2 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/DCA/DCA_hnsc_prediction.csv', index_col=0)
CpG_csv2 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/CpG/CpG_hnsc_prediction.csv', index_col=0)
scVI_csv2 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/scVI/scVI_hnsc_prediction.csv', index_col=0)
DeepMicroGen_csv2 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/DeepMicroGen/DeepMicroGen_hnsc_prediction.csv', index_col=0)

data_list = [VDiT_csv1, VDiT_csv2, DeepImpute_csv2, AutoImpute_csv2, DCA_csv2, CpG_csv2, scVI_csv2, DeepMicroGen_csv2]
i = 0

for data in data_list:
    map(VDiT_csv1, data, i)
    i = i + 1
'''


def pcc_3(csv1, csv2):
    # 获取列数（假设两个文件的列数相同）
    num_columns = len(csv1.columns)

    # 初始化 PCC 结果列表
    pcc_results = []

    # 计算每一列的 PCC
    for i in range(num_columns):
        pcc = csv1.iloc[:, i].corr(csv2.iloc[:, i])
        pcc_results.append(pcc)

    return pcc_results

VDiT_csv1 = pd.read_csv('D:\\PyCharm\\Py_Projects\\DiTT\\result\\data_hnsc\\original.csv', index_col=0)
VDiT_csv2 = pd.read_csv('D:\\PyCharm\\Py_Projects\\DiTT\\result\\data_hnsc\\prediction.csv', index_col=0)

DeepImpute_csv1 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/DeepImpute/DeepImpute_hnsc_origin.csv', index_col=0)
DeepImpute_csv2 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/DeepImpute/DeepImpute_hnsc_prediction.csv', index_col=0)

AutoImpute_csv1 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/AutoImpute/AutoImpute_hnsc_origin.csv', index_col=0)
AutoImpute_csv2 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/AutoImpute/AutoImpute_hnsc_prediction.csv', index_col=0)

DCA_csv1 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/DCA/DCA_hnsc_origin.csv', index_col=0)
DCA_csv2 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/DCA/DCA_hnsc_prediction.csv', index_col=0)

CpG_csv1 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/CpG/CpG_hnsc_origin.csv', index_col=0)
CpG_csv2 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/CpG/CpG_hnsc_prediction.csv', index_col=0)

scVI_csv1 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/CpG/CpG_hnsc_origin.csv', index_col=0)
scVI_csv2 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/scVI/scVI_hnsc_prediction.csv', index_col=0)

DeepMicroGen_csv1 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/DeepMicroGen/DeepMicroGen_hnsc_origin.csv', index_col=0)
DeepMicroGen_csv2 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/DeepMicroGen/DeepMicroGen_hnsc_prediction.csv', index_col=0)

data_list = [[VDiT_csv1, VDiT_csv2], [DeepImpute_csv1, DeepImpute_csv2], [AutoImpute_csv1, AutoImpute_csv2],
             [DCA_csv1, DCA_csv2], [CpG_csv1, CpG_csv2], [DeepMicroGen_csv1, DeepMicroGen_csv2], [scVI_csv1, scVI_csv2]]

pcc_results = []

for data in data_list:
    pcc_results.append(pcc_3(data[0], data[1]))



import numpy as np
t = pcc_results[0]
for i in range(1, len(pcc_results)):
    t1 = np.array(pcc_results[i])
    t = np.vstack((t, t1))
data_map = t.T

sorted_lists = sorted(zip(t[0], t[1], t[2], t[3], t[4], t[5], t[6]), reverse=True)
s0, s1, s2, s3, s4, s5, s6 = zip(*sorted_lists)
tuples = [s0, s1, s2, s3, s4, s5, s6]
tuples = np.array(tuples)
sns.heatmap(tuples.T, cmap=blue_gold_cmap, cbar=True, annot=False)

#sns.heatmap(data_map, cmap=blue_gold_cmap, cbar=True, annot=False)
plt.savefig('D:\TCGA\mbVDiT_data\WXS\solid\hnsc\Fig4\heatmap.pdf', format='pdf', bbox_inches='tight')
plt.show() #0:VDiT 1:DeepImpute 2:AutoImpute 3:DCA 4:DeepMicroGen 5:CpG 6:scVI
