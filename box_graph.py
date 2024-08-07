import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def shannon_index(relative_abundance):
    """
    计算Shannon指数
    Args:
        relative_abundance: 一维张量，表示每个物种的相对丰度
    Returns:
        Shannon指数
    """
    normalized_abundance = relative_abundance / torch.sum(relative_abundance)
    # 计算每个物种的相对丰度的对数
    log_relative_abundance = torch.log(normalized_abundance + 1e-8)  # 避免log(0)的情况出现

    # 计算Shannon指数
    shannon = -torch.sum(normalized_abundance * log_relative_abundance)

    return shannon


# 读取CSV文件
data1 = pd.read_csv('D:\\PyCharm\\Py_Projects\\DiTT\\result\\data_hnsc\\original.csv', index_col=0)
# 计算Shannon指数并添加到DataFrame中
print(data1.shape)
data1['Shannon_index'] = data1.apply(lambda row: shannon_index(torch.tensor(row)), axis=1)
test1 = data1['Shannon_index']
t1 = list(test1)

data2 = pd.read_csv('D:\\PyCharm\\Py_Projects\\DiTT\\result\\data_hnsc\\prediction.csv', index_col=0)
# 计算Shannon指数并添加到DataFrame中
print(data2.shape)
data2['Shannon_index'] = data2.apply(lambda row: shannon_index(torch.tensor(row)), axis=1)
test2 = data2['Shannon_index']
t2 = list(test2)

data3 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/DeepImpute/DeepImpute_hnsc_prediction.csv', index_col=0)
# 计算Shannon指数并添加到DataFrame中
print(data3.shape)
data3['Shannon_index'] = data3.apply(lambda row: shannon_index(torch.tensor(row)), axis=1)
test3 = data3['Shannon_index']
t3 = list(test3)

data4 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/AutoImpute/AutoImpute_hnsc_prediction.csv', index_col=0)
# 计算Shannon指数并添加到DataFrame中
print(data4.shape)
data4['Shannon_index'] = data4.apply(lambda row: shannon_index(torch.tensor(row)), axis=1)
test4 = data4['Shannon_index']
t4 = list(test4)

data5 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/DCA/DCA_hnsc_prediction.csv', index_col=0)
# 计算Shannon指数并添加到DataFrame中
print(data5.shape)
data5['Shannon_index'] = data5.apply(lambda row: shannon_index(torch.tensor(row)), axis=1)
test5 = data5['Shannon_index']
t5 = list(test5)

data6 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/CpG/CpG_hnsc_prediction.csv', index_col=0)
# 计算Shannon指数并添加到DataFrame中
print(data6.shape)
data6['Shannon_index'] = data6.apply(lambda row: shannon_index(torch.tensor(row)), axis=1)
test6 = data6['Shannon_index']
t6 = list(test6)

data7 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/scVI/scVI_hnsc_prediction.csv', index_col=0)
# 计算Shannon指数并添加到DataFrame中
print(data7.shape)
data7['Shannon_index'] = data7.apply(lambda row: shannon_index(torch.tensor(row)), axis=1)
test7 = data7['Shannon_index']
t7 = list(test7)

data8 = pd.read_csv('D:/PyCharm/Py_Projects/Diffusion/baseline_result/DeepMicroGen/DeepMicroGen_hnsc_prediction.csv', index_col=0)
# 计算Shannon指数并添加到DataFrame中
print(data8.shape)
data8['Shannon_index'] = data8.apply(lambda row: shannon_index(torch.tensor(row)), axis=1)
test8 = data8['Shannon_index']
t8 = list(test8)

data = [np.array(t1), np.array(t2), np.array(t8), np.array(t3), np.array(t4), np.array(t5), np.array(t6), np.array(t7)]
ax = sns.boxplot(data=data, orient='v', palette='Set2', width=0.3)
sns.set(style="whitegrid")
plt.xlabel('Method')
plt.ylabel('Shannon Index')
plt.title('Dataset_STAD')
plt.xticks(range(len(data)), list(['Real samples', 'VDiT', 'DeepMicroGen', 'DeepImpute', 'AutoImpute', 'DCA', 'CpG', 'scVI']))
plt.savefig('D:\\TCGA\mbVDiT_data\\WXS\\solid\\hnsc\\Fig3\\HNSC.pdf', format='pdf', bbox_inches='tight')
plt.show()

# 打印结果
print()