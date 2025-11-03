import numpy as np

# 加载所有 CSV 文件
file_names = [
    "deepDDI/class_metrics_fold_0.csv",
    "deepDDI/class_metrics_fold_1.csv",
    "deepDDI/class_metrics_fold_2.csv",
    "deepDDI/class_metrics_fold_3.csv",
    "deepDDI/class_metrics_fold_4.csv"
]

# 加载数据，delimiter="," 参数指定了CSV文件的分隔符为逗号。
data_list = [np.loadtxt(file, delimiter=",", skiprows=1) for file in file_names]

# 将所有数据堆叠成一个三维数组，每个文件的数据成为数组的一个“层”。
data_stack = np.stack(data_list)

# 计算均值
mean_data = np.max(data_stack, axis=0)

# 保存为新的 CSV 文件
np.savetxt("deepDDI/mean_metrics_max.csv", mean_data, delimiter=",")

