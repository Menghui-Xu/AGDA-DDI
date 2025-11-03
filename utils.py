from __future__ import print_function

import scipy.sparse as sp  #用于处理稀疏矩阵
import numpy as np
import networkx as nx  #用于创建和操作图
import os
import pandas as pd
import torch
from sklearn.decomposition import PCA # 用于主成分分析

#定义归一化函数，输入的邻接矩阵，可以是稀疏矩阵或密集矩阵。指定是否进行对称归一化，默认为 True。
def normalize_adj(adj, symmetric=True):
    #进行对称归一化
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()  #使用 tocsr 方法将结果转换为压缩稀疏行（CSR）格式，这是一种高效的稀疏矩阵存储格式
    #非对称归一化
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm



#邻接矩阵 adj 中添加单位矩阵（自连接），并归一化
def preprocess_adj(adj, symmetric=True):
    # 在邻接矩阵 adj 中添加单位矩阵
    adj = adj + sp.eye(adj.shape[0])
    # 归一化处理
    adj = normalize_adj(adj, symmetric)
    return adj



#创建一个掩码，用于从图中选择特定的节点或样本，idx：一个索引数组，表示要选择的样本的索引，l图中节点的总数
def sample_mask(idx, l):
    mask = np.zeros(l)  #创建一个长度为 l 的全零数组 mask
    mask[idx] = 1
    return np.array(mask, dtype=np.bool) #将掩码数组转换为布尔类型



#用于将标签数据 y 分割成训练集、验证集和测试集，还生成了一个训练集掩码
def get_splits(y):

    idx_train = range(140)          # 训练集索引：0-139
    idx_val = range(200, 500)       # 验证集索引：200-499
    idx_test = range(500, 1500)     # 测试集索引：500-1499
    #创建三个与 y 形状相同、类型为零数组，分别用于存储训练集、验证集和测试集的标签。
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    #将原始标签 y 中对应索引范围的标签分别赋值给 y_train、y_val 和 y_test
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    #调用 sample_mask 函数生成训练集的掩码，掩码是一个布尔数组，长度与 y 相同，其中训练集索引位置为 True，其余位置为 False。
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


# def get_adj_mat():
#
#     edge_df = pd.read_csv('ddi_class_65.csv')
#     G = nx.from_pandas_edgelist(edge_df,source='Drug1',target='Drug2')
#     drug2id = {}
#     nodes = list(G.nodes)
#     total_num_nodes = G.number_of_nodes()
#     for i,drug in enumerate(nodes):
#         drug2id[drug] = i
#     nx.relabel_nodes(G,drug2id,copy=False)
#
#     edges = list(G.edges)
#     edges = np.array(edges)
#
#     return edges,total_num_nodes


#读取边数据，构建一个无向图，并提取图的边列表和节点总数
def get_adj_mat():
    # 读取边数据
    #edge_df = pd.read_csv('Deng/ddi_class_65.csv')
    edge_df = pd.read_csv('deepDDI/KnownDDI.csv')

    # 构建图  Drug1--》Drug2
    G = nx.from_pandas_edgelist(edge_df, source='Drug1', target='Drug2', create_using=nx.Graph())


    # 检查节点是否已经是整数 ID
    nodes = list(G.nodes)
    if not all(isinstance(node, int) for node in nodes):
        raise ValueError("Nodes are not integer IDs. Please check the Deng.")

    # 提取边列表并转换为 NumPy 数组
    edges = list(G.edges)
    edges = np.array(edges, dtype=np.int32)  # 确保边列表是整数类型

    # 获取节点总数
    total_num_nodes = G.number_of_nodes()
    return edges, total_num_nodes


#从指定的文件中加载特征数据，并将这些数据转换为 PyTorch 张量,similarity_profile_file：输入的特征文件路径
def load_feat(similarity_profile_file):
    #读取特征数据
    init_feat = pd.read_csv(similarity_profile_file, header=None)
    #转换为NumPy数组
    init_feat = init_feat.to_numpy()
    # pca = PCA(0.99)
    # pca = PCA(512)
    # feat_mat = pca.fit_transform(init_feat)
    return torch.FloatTensor(init_feat), torch.FloatTensor(init_feat)#返回两个相同的PyTorch浮点张量



#从给定的边数据中分割训练集和测试集，并构建训练集的邻接矩阵,ratio：用于分割测试集的比例，默认为0.2
def get_train_test_set(edges_all, num_nodes, ratio=0.2):
    # 打乱和分割边数据
    np.random.shuffle(edges_all)
    test_size = int(edges_all.shape[0] * ratio)
    test_edges_true = edges_all[0:test_size]
    train_edges_true = edges_all[test_size:]

    # 读取和分割负样本
    df = pd.read_csv('./negative_samples.csv', index_col=0)
    samples = df.to_numpy()
    np.random.shuffle(samples)
    test_false_size = int(samples.shape[0] * ratio)
    test_edges_false = samples[:test_false_size, :-1].astype(np.int32)
    train_edges_false = samples[test_false_size:, :-1].astype(np.int32)

    #构建训练集邻接矩阵
    data = np.ones(train_edges_true.shape[0])  #创建一个与训练集真边数量相同的全1数组 Deng
    # Re-build adj matrix
    #构建训练集的邻接矩阵 adj_train，其中只包含单向边。
    adj_train = sp.csr_matrix((data, (train_edges_true[:, 0], train_edges_true[:, 1])), shape=(num_nodes, num_nodes),dtype=np.float32)
    adj_train = adj_train + adj_train.T #将 adj_train 与其转置相加，以确保邻接矩阵是对称的（无向图）。

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges_true, train_edges_false, test_edges_true, test_edges_false


#创建一个完整的数据集
def get_full_dataset():
    edge_df = pd.read_csv('Deng/ddi_class_65.csv')
    G = nx.from_pandas_edgelist(edge_df, source='Drug1', target='Drug2')

    drug2id = {} #创建一个字典 drug2id 来存储药物名称到节点ID的映射
    nodes = list(G.nodes)
    for i, drug in enumerate(nodes): #遍历图中的节点，并使用枚举为每个节点分配一个唯一的整数ID。
        drug2id[drug] = i
    nx.relabel_nodes(G, drug2id, copy=False) #根据 drug2id 字典重新标记图中的节点。


    edges_all = list(G.edges)
    edges_all = np.array(edges_all)
    np.random.shuffle(edges_all)
    y_true = np.ones(edges_all.shape[0], dtype=np.int8)

    df = pd.read_csv('./negative_samples.csv', index_col=0)
    samples = df.to_numpy()
    np.random.shuffle(samples)
    false_edges = samples[:, :-1]
    y_false = np.zeros(false_edges.shape[0], dtype=np.int8)

    return np.vstack([edges_all, false_edges]), np.concatenate([y_true, y_false])
