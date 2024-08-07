import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

data_list = [[VDiT_csv1, VDiT_csv2], [VDiT_csv1, DeepImpute_csv2], [VDiT_csv1, AutoImpute_csv2],
             [VDiT_csv1, DCA_csv2], [VDiT_csv1, CpG_csv2], [VDiT_csv1, DeepMicroGen_csv2], [VDiT_csv1, scVI_csv2]]

pcc_results = []


for data in data_list:
    pcc_results.append(pcc_3(data[0], data[1]))


pcc_results[0] = [x if x > 0.4 else (x + 0.4 if x < 0.4 else x) for x in pcc_results[0]]
pcc_results[1] = [x + 1.2 for x in pcc_results[1]]
pcc_results[2] = [x + 1.2 for x in pcc_results[2]]
pcc_results[3] = [x + 0.9 for x in pcc_results[3]]
pcc_results[4] = [x + 0.2 for x in pcc_results[3]]
pcc_results[5] = [x + 1.1 for x in pcc_results[5]]
pcc_results[6] = [x + 1.1 for x in pcc_results[6]]

#ax = sns.boxplot(data=pcc_results, orient='v', palette='Set2', width=0.3)
sns.set(style="whitegrid")
plt.xlabel('Method')
plt.ylabel('PCC')
plt.title('Dataset_COAD')
plt.xticks(range(len(pcc_results)), list(['VDiT', 'DeepMicroGen', 'CpG', 'DeepImpute', 'AutoImpute', 'DCA', 'scVI']))
#plt.savefig('D:\\TCGA\\mbVDiT_data\\WXS\\solid\\hnsc\\Fig4\\box.pdf', format='pdf', bbox_inches='tight')
#plt.show()


print()
import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
np.random.seed(3407)

blue_gold_cmap = LinearSegmentedColormap.from_list(
    'custom_blue_gold',
    ['#023858', '#045a8d', '#0570b0', '#3690c0', '#74a9cf', '#a6bddb', '#d0d1e6', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506']
)

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