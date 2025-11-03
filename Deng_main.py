# -*- coding: utf-8 -*-
# import openTSNE

import os, pickle, glob
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.inspection import permutation_importance
from PIL import Image
import shutil

import os, pandas as pd, numpy as np, umap, sklearn.manifold
from sklearn.decomposition import PCA
import os, pickle
import numpy
import argparse
import csv
import time
from collections import defaultdict
# 字典的子类，自动为不存在的键生成默认值（如 defaultdict(list) 初始化为空列表）
from functools import reduce
# 对序列中的元素累积应用函数（如求和 reduce(lambda x, y: x+y, [1, 2, 3]) 返回 6）
import numpy as np
import pandas as pd
import torch
# 对序列中的元素累积应用函数（如求和 reduce(lambda x, y: x+y, [1, 2, 3]) 返回 6）
import torch.nn as nn
# 神经网络模块的基类（如 nn.Module）、层（如 nn.Linear）、损失函数（如 nn.CrossEntropyLoss）
import torch.nn.functional as F
# 函数式接口（无状态），提供激活函数（如 F.relu）、卷积操作等
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, f1_score, precision_score, recall_score
# 模型评估指标，如 roc_auc_score（AUC值）、precision_recall_curve（PR曲线）、f1_score（F1分数）
from sklearn.preprocessing import label_binarize
# 数据预处理，如 label_binarize 将多类标签转为二值矩阵
from torch.utils.data import Dataset, DataLoader
from Deng_model import Model, get_edge_index, Model_avg, Model_sum, Model_wo_feat, Model_wo_diffgraph, Model_wo_relgraph, \
    Model_wo_att, Model_GAT, Model_GCN, Model_cat, Model_hdm
from utils import get_adj_mat, load_feat
# get_adj_mat生成邻接矩阵



# '''Code for multi-class DDI prediction'''
# # 创建了一个 ArgumentParser 对象，用于处理命令行参数。
# parser = argparse.ArgumentParser()
# parser.add_argument('--hidden_dim', type=int, default=512, help='num of hidden features')##隐藏层的维度，即神经网络中隐藏层的节点数。
# parser.add_argument('--out_dim', type=int, default=256, help='num of output features')#输出层的维度，即模型输出的特征数。
# parser.add_argument('--dropout', type=float, default=0.4, help='dropout rate')#Dropout 率，用于防止过拟合的一种正则化技术，通过随机丢弃一部分神经元来实现。
# parser.add_argument('--rels', type=int, default=65, help='num of relations')#关系的数量，可能用于定义模型中的关系类型或类别数
# parser.add_argument('--n_epochs', type=int, default=400, help='num of epochs')#训练的轮数，即整个训练数据集被遍历的次数。
# parser.add_argument('--batch_size', type=int, default=1024, help='batch size')#批量大小，即每次训练时使用的样本数量
# parser.add_argument('--threshold', type=float, default=0.4, help='edge threshold')#边的阈值，可能用于确定哪些边应该被添加到图中。
# parser.add_argument('--model',default='1', type=str)#模型的类型，用于选择不同的模型架构
# args = parser.parse_args()#解析命令行输入的参数，并将它们存储在 args 对象中，以便在脚本的其他部分使用。
#

'''Code for multi-class DDI prediction'''
# 创建了一个 ArgumentParser 对象，用于处理命令行参数。
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0,help='which GPU card to use, 0/1/2/3 …')#服务器第几张卡
parser.add_argument('--hidden_dim', type=int, default=512, help='num of hidden features')##隐藏层的维度，即神经网络中隐藏层的节点数。
parser.add_argument('--out_dim', type=int, default=256, help='num of output features')#输出层的维度，即模型输出的特征数。
parser.add_argument('--dropout', type=float, default=0.4, help='dropout rate')#Dropout 率，用于防止过拟合的一种正则化技术，通过随机丢弃一部分神经元来实现。
parser.add_argument('--rels', type=int, default=65, help='num of relations')#关系的数量，可能用于定义模型中的关系类型或类别数
# parser.add_argument('--rels', type=int, default=86, help='num of relations')#关系的数量，可能用于定义模型中的关系类型或类别数
parser.add_argument('--n_epochs', type=int, default=400, help='num of epochs')#训练的轮数，即整个训练数据集被遍历的次数。
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')#批量大小，即每次训练时使用的样本数量
parser.add_argument('--threshold', type=float, default=0.4, help='edge threshold')#边的阈值，可能用于确定哪些边应该被添加到图中。
parser.add_argument('--model',default='1', type=str)#模型的类型，用于选择不同的模型架构
parser.add_argument('--filename', type=str, default='result.csv', help='folder to save results.csv')
args = parser.parse_args()#解析命令行输入的参数，并将它们存储在 args 对象中，以便在脚本的其他部分使用。

hidden_dim = args.hidden_dim
out_dim = args.out_dim
n_epochs = args.n_epochs
batch_size = args.batch_size


# if torch.cuda.device_count() >= 1:
#     device = torch.device("cuda:0")  # 使用第二张卡，索引为1
#     print("use cuda:0")
# else:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 把原来的 device 选择逻辑改成
device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] using device: {device}')
print(f"数据保存文件为：{args.filename}")



# 将一个 SciPy 稀疏矩阵转换为 PyTorch 的稀疏张量
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


edges_all, num_nodes = get_adj_mat()

# 数据读取与分层抽样
def get_straitified_data(ratio=0.2):
    print("测试集比例",ratio)
    # df_data = pd.read_csv('Deng/old_KnownDDI.csv')
    df_data = pd.read_csv('Deng/ddi_class_65.csv')
    all_tup = [(h, t, r) for h, t, r in zip(df_data['Drug1'], df_data['Drug2'], df_data['Label'])]#从数据框中提取 Drug1、Drug2 和 Label 列，并生成一个包含所有样本的元组列表 all_tup。
    np.random.shuffle(all_tup) #随机打乱样本元组列表，以确保数据的随机性。
    tuple_by_type = defaultdict(list)
    for h, t, r in all_tup:
        tuple_by_type[r].append((h, t, r))#根据标签 r 将样本元组分组，并存储在 defaultdict 中。
    tuple_by_type.keys().__len__()
    train_edges = []
    test_edges = []
    for r in tuple_by_type.keys(): #对每个类别进行分层抽样，将前 ratio 比例的样本分配到测试集，其余样本分配到训练集。
        test_edges.append(tuple_by_type[r][:int(len(tuple_by_type[r]) * ratio)])
        train_edges.append(tuple_by_type[r][int(len(tuple_by_type[r]) * ratio):])

    def merge(x, y):
        return x + y

    test_tups = np.array(reduce(merge, test_edges))

    train_tups = np.array(reduce(merge, train_edges))

    train_edges = train_tups[:, :2]
    train_labels = train_tups[:, -1]
    test_edges = test_tups[:, :2]
    test_labels = test_tups[:, -1]
    print("测试集边数",len(test_edges))
    return train_edges, train_labels, test_edges, test_labels


# testset_ratio = 0.2
testset_ratio = 0.4
train_edges, train_labels, test_edges, test_labels = get_straitified_data(testset_ratio)  # Get training and testing sets
# feat_mat = load_feat('Deng//drug_sim.csv')  #Load drug attribute file
feat_mat_chem, sim_mat_chem = load_feat('Deng/chem_Jacarrd_sim.csv')  # Load drug attribute file
feat_mat_target, sim_mat_target = load_feat('Deng/target_Jacarrd_sim.csv')  # Load drug attribute file
feat_mat_enzyme, sim_mat_enzyme = load_feat('Deng/enzyme_Jacarrd_sim.csv')  # Load drug attribute file
feat_mat_pathway, sim_mat_pathway = load_feat('Deng/pathway_Jacarrd_sim.csv')  # Load drug attribute file
train_feat_mat = torch.hstack([feat_mat_chem, feat_mat_target, feat_mat_enzyme, feat_mat_pathway])#水平堆叠所有特征矩阵，形成一个综合的特征矩阵 train_feat_mat。
train_feat_mat = torch.FloatTensor(train_feat_mat)
train_feat_mat = train_feat_mat.to(device)
args.feat_dim = train_feat_mat.shape[1]#更新 args 对象中的 feat_dim 参数，以便在模型定义中使用特征维度。

# ====== 自动获取四类属性的真实维度 ======
feat_mat_chem, _   = load_feat('Deng/chem_Jacarrd_sim.csv')
feat_mat_target, _ = load_feat('Deng/target_Jacarrd_sim.csv')
feat_mat_enzyme, _ = load_feat('Deng/enzyme_Jacarrd_sim.csv')
feat_mat_pathway, _= load_feat('Deng/pathway_Jacarrd_sim.csv')
attr_dims = [feat_mat_chem.shape[1], feat_mat_target.shape[1],
             feat_mat_enzyme.shape[1], feat_mat_pathway.shape[1]]
print('[INFO] 四类属性维度:', attr_dims)     # 例如 [756, 636, 445, 327]


#根据相似度矩阵生成新的边
def generate_edges_from_similarity(train_edges,  similarity_matrix,  threshold):
    """
    根据相似度矩阵生成新的边，且新增的边只能包含训练集中出现的药物节点

    参数:
    - similarity_matrix: 相似度矩阵，形状为 (num_nodes, num_nodes)
    - threshold: 相似度阈值，超过这个值的节点对将被连接
    - train_edges: 训练集中的边列表，形状为 (num_train_edges, 2)

    返回:
    - new_edges: 新的边列表，形状为 (num_new_edges, 2)
    """
    # 从 train_edges 中提取所有训练集节点
    train_nodes = np.unique(train_edges.flatten()) #提取所有唯一的节点，并存储在 train_nodes 数组中。
    num_nodes = similarity_matrix.shape[0]
    new_edges = [] #初始化一个空列表 new_edges，用于存储新生成的边。

    train_nodes_set = set(train_nodes) #将训练集节点转换为集合，方便快速查找

    for i in range(num_nodes):
        # 仅处理训练集中的节点
        if i not in train_nodes_set:
            continue
        for j in range(i + 1, num_nodes):
            # 仅处理训练集中的节点
            if j not in train_nodes_set:
                continue
            if similarity_matrix[i, j] > threshold:#相似度超过阈值，则将这个节点对添加到 new_edges 列表中。
                new_edges.append([i, j])

    return np.array(new_edges)#将 new_edges 列表转换为一个 NumPy 数组，并返回这个数组。


if args.threshold == 1:
    new_train_edges = train_edges
else:
    new_edges_chem = generate_edges_from_similarity(train_edges, sim_mat_chem.cpu().numpy(), threshold=args.threshold)
    new_edges_target = generate_edges_from_similarity(train_edges, sim_mat_target.cpu().numpy(),
                                                      threshold=args.threshold)
    new_edges_enzyme = generate_edges_from_similarity(train_edges, sim_mat_enzyme.cpu().numpy(),
                                                      threshold=args.threshold)
    new_edges_pathway = generate_edges_from_similarity(train_edges, sim_mat_pathway.cpu().numpy(),
                                                       threshold=args.threshold)
    #
    # print(len(new_edges_chem),len(new_edges_target),len(new_edges_enzyme),len(new_edges_pathway))
    # print(len(train_edges))
    # # 合并所有新的边
    new_train_edges = np.vstack([train_edges, new_edges_chem, new_edges_target, new_edges_enzyme, new_edges_pathway])
    # new_train_edges = train_edges

    # 去除重复的边
    new_train_edges = np.unique(new_train_edges, axis=0)
    # print(len(new_train_edges))



class DDIDataset(Dataset):
    '''Customized dataset processing class'''

    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


train_dataset = DDIDataset(train_edges, train_labels)#使用自定义的 DDIDataset 类创建训练数据集和测试数据集。
test_dataset = DDIDataset(test_edges, test_labels)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)#DataLoader：PyTorch 提供的数据加载器，用于批量加载数据。
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)


#根据命令行参数 args.model 的值选择不同的模型架构
if args.model == '1':
    model = Model(args)
if args.model == 'sum':
    model = Model_sum(args)
if args.model == 'cat':
    model = Model_cat(args)
if args.model == 'avg':
    model = Model_avg(args)
if args.model == 'hdm':
    model = Model_hdm(args)
if args.model == 'feat':
    model = Model_wo_feat(args)
if args.model == 'dg':
    model = Model_wo_diffgraph(args)
if args.model == 'rg':
    model = Model_wo_relgraph(args)
if args.model == 'att':
    model = Model_wo_att(args)
if args.model == 'gat':
    model = Model_GAT(args)
if args.model == 'gcn':
    model = Model_GCN(args)

#将模型移动到之前设置的设备（CPU或GPU）上。
model.to(device)





def train_():
    criterion = nn.CrossEntropyLoss().to(device) #交叉熵损失函数，用于多分类问题
    re_criterion = nn.MSELoss().to(device)  #均方误差损失函数，评估重构输出（如自编码器）与原始输入的误差

    trainer_E = torch.optim.Adam(model.parameters(), lr=1e-3) #使用 Adam 优化器来更新模型参数，学习率为 1e-3

    n_iterations = len(train_loader) #每个 epoch 中的迭代次数，等于训练数据加载器的长度。
    num_epochs = n_epochs #总的训练轮数
    start = time.time() #记录训练开始时间

    running_loss = 0.0
    running_correct = 0.0

    # ========== 1. 统一全采样索引 ==========
    idx = np.arange(len(test_edges))  # 全采样
    np.random.seed(42)  # 保证复现

    # ========== 2. 保存原始特征 UMAP ==========
    save_raw_edge_umap_fixed(idx)



    # Training phase
    for epoch in range(num_epochs):
        model.train()
        true_labels, pred_labels = [], []
        running_loss = 0.0 #累计损失
        running_correct = 0.0 #累计正确分类的数量
        total_samples = 0 #总样本数
        for i, (edges, labels) in enumerate(train_loader):
            edges, labels = edges.to(device), labels.to(device)  # 迁移到GPU

            # edge_index：主图的邻接关系，diff_edge_index：增广图的邻接关系，edge_type：边的类型信息，train_feat_mat：节点特征矩阵，edges：当前批次的边数据
            edge_index, edge_type = get_edge_index(train_edges, train_labels)#构建图的邻接矩阵（可能包含边类型信息）
            print(edge_index.shape)
            diff_edge_index, _ = get_edge_index(new_train_edges, train_labels)#构建图的邻接矩阵（可能包含边类型信息）

            edge_index, edge_type, diff_edge_index = edge_index.to(device), edge_type.to(device), diff_edge_index.to(device)

            #y_pred：分类任务的预测（未归一化的概率），re_out：重构输出
            #print(edge_index[:10], diff_edge_index[:10], edge_type[:10], train_feat_mat[:10], edges[:10])
            y_pred, re_out = model(edge_index, diff_edge_index, edge_type, train_feat_mat, edges)


            # y_pred = model(edge_index, diff_edge_index, edge_type, train_feat_mat, edges)
            loss = criterion(y_pred, labels)
            re_loss = re_criterion(re_out, train_feat_mat)
            loss += re_loss

            trainer_E.zero_grad() # 清空历史梯度
            loss.backward() #通过计算图反向传播梯度（自动微分）
            trainer_E.step() #根据梯度方向更新参数（Adam自适应调整学习率）


            #torch.argmax(y_pred, dim=1)：找到概率最大的类别索引，即预测的类别。
            #torch.sum(...)：计算预测正确的样本总数，running_correct：累计正确分类的数量。
            running_correct += torch.sum((torch.argmax(y_pred, dim=1).type(torch.FloatTensor) == labels.cpu()).detach()).float()

            #将当前批次的预测标签添加到 pred_labels 列表中。
            pred_labels.append(list(y_pred.cpu().detach().numpy().reshape(-1)))

            labels = labels.cpu().numpy()
            #将当前批次的真实标签添加到 true_labels 列表中。
            true_labels.append(list(labels))

            #labels.shape[0]：获取当前批次的样本数量，total_samples += ...：累计总样本数量
            total_samples += labels.shape[0]
            #获取当前批次的损失值，并累计总损失
            running_loss += loss.item()

        print(
            f"epoch {epoch + 1}/{num_epochs};trainging loss: {running_loss / n_iterations:.4f} ;training set acc: {running_correct / total_samples:.4f}")
        if (epoch + 1) % 50 == 0:

            print('---- 开始可视化 ----')
            save_viz_csv_fixed(idx, model, test_edges[idx], test_labels[idx], device, epoch=epoch + 1)

            print('---- 可视化完成 ----')

            # test_()

    end = time.time() #结束训练的时间
    elapsed = end - start #训练时间统计
    print(f"Training completed in {elapsed // 60}m: {elapsed % 60:.2f}s.")




# y_true = np.array([[0, 1], [1, 0], [0, 1]])  # 2个类别的真实标签
# y_score = np.array([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]) # 预测概率
#计算多类分类问题中每个类别的ROC-AUC和PR-AUC（平均精度）分数
def roc_aupr_score(y_true, y_score, average="macro"):
    #计算二分类问题的ROC-AUC和PR-AUC分数
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    #计算多类分类问题的平均分数。
    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()

        # 确保输入为二维数组（n_samples × n_classes）
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))

        # 逐类别计算AUPR
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()  # 取第c类的真实标签
            y_score_c = y_score.take([c], axis=1).ravel()  # 取第c类的预测概率
            score[c] = binary_metric(y_true_c, y_score_c)    #计算该类AUPR
        return np.average(score)
    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)




def test_():
    event_num = 65         # 类别的数量，这里假设有65个不同的事件或类别
    n_test_samples = 0     # 测试样本的总数
    n_correct = 0          # 正确预测的样本数
    total_labels = []      # 存储所有测试样本的真实标签
    total_pred = []        # 存储所有测试样本的预测概率
    # Testing phase
    with torch.no_grad():  # 确保在评估过程中不计算梯度，节省计算资源。
        model.eval()       # 将模型设置为评估模式
        for edges, labels in test_loader:

            ## 获取图结构数据并移至GPU
            # edge_index：主图的邻接关系，diff_edge_index：增广图的邻接关系，edge_type：边的类型信息，train_feat_mat：节点特征矩阵，edges：当前批次的边数据
            edge_index, edge_type = get_edge_index(train_edges, train_labels)
            diff_edge_index, _ = get_edge_index(new_train_edges, train_labels)

            edge_index, edge_type, diff_edge_index = edge_index.to(device), edge_type.to(device), diff_edge_index.to(device)
            edges, labels = edges.to(device), labels.to(device)

            ## 前向传播（仅计算分类输出） y_pred：分类任务的预测（未归一化的概率）
            y_pred, _ = model(edge_index, diff_edge_index, edge_type, train_feat_mat, edges)
            # y_pred= model(edge_index, diff_edge_index, edge_type, train_feat_mat, edges)
            #使用 softmax 函数将预测结果转换为概率分布
            y_hat = F.softmax(y_pred, dim=1)


            #将预测概率和真实标签从设备移动到CPU，并转换为NumPy数组。
            total_pred.append(y_hat.cpu().numpy())
            total_labels.append(labels.cpu().numpy())

            #计算测试样本总数，正确预测数
            n_test_samples += edges.shape[0]
            n_correct += torch.sum((torch.argmax(y_pred, dim=1).type(torch.FloatTensor) == labels.cpu()).detach()).float()


        total_pred = np.vstack(total_pred)  # 合并所有批次的预测概率
        total_labels = np.concatenate(total_labels)  # 合并真实标签
        pred_type = np.argmax(total_pred, axis=1)  # 获取预测类别 ([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]) # 预测概率
        y_one_hot = label_binarize(y=total_labels, classes=np.arange(event_num))  #([[0, 1], [1, 0], [0, 1]])  # 2个类别的真实标签


        acc = n_correct / n_test_samples  # 计算总体准确率。
        # y_true = np.array([[0, 1], [1, 0], [0, 1]])  # 2个类别的真实标签
        # y_score = np.array([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]) # 预测概率
        aupr = roc_aupr_score(y_one_hot, total_pred, average='micro')
        auroc = roc_auc_score(y_one_hot, total_pred, average='micro')
        f1 = f1_score(total_labels, pred_type, average='macro')
        precision = precision_score(total_labels, pred_type, average='macro', zero_division=0)
        recall = recall_score(total_labels, pred_type, average='macro', zero_division=0)

        print(f"-------test set result---------")
        print(f"ACC: {acc}")
        print(f"AUPR: {aupr}")
        print(f"AUROC: {auroc}")
        print(f"F1: {f1}")
        print(f"Precison: {precision}")
        print(f"Recall: {recall}")
        # print(f"{acc}\t{aupr}\t{auroc}{f1}\t{precision}\t{recall}")
        save_results_to_file(args.model, args.threshold, args.dropout, args.n_epochs, acc, aupr, auroc, f1, precision, recall)
        # 案例分析数据
        # save_case_study(test_edges, total_labels, total_pred, train_edges)


def save_results_to_file(t_model, t_threshold, t_dropout, t_n_epochs, acc, aupr, auroc, f1, precision, recall, filename=args.filename):
    # 使用 try-except 结构检查文件是否存在。如果文件不存在，写入表头
    try:
        with open(filename, 'r') as f:
            pass
    except FileNotFoundError: ## 文件不存在时创建
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'threshold', 'dropout', 'n_epochs', 'threshold', 'ACC', 'AUPR', 'AUROC', 'F1', 'Precision', 'Recall'])

    #以追加模式（'a'）写入数据。
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([t_model, t_threshold, t_dropout, t_n_epochs,acc, aupr, auroc, f1, precision, recall])



def save_case_study(test_edges, total_labels, total_pred, train_edges,
                       topk_class=10, topk_global=20, out_dir='case_data'):
    """
    65 类专用：保存
    1) 每类 topk_class 高置信样本
    2) 全局 topk_global 高置信样本（csv）
    3) zero-shot 数据
    """
    os.makedirs(out_dir, exist_ok=True)
    event_num = 65

    # 1. 每类 topk
    high_conf = {}
    global_buf = []
    for c in range(event_num):
        idx_c = np.where(total_labels == c)[0]
        if len(idx_c) == 0:
            continue
        prob_c = total_pred[idx_c, c]
        top_idx = idx_c[prob_c.argsort()[-min(topk_class, len(idx_c)):][::-1]]
        high_conf[c] = [(int(test_edges[i, 0]),
                         int(test_edges[i, 1]),
                         float(total_pred[i, c])) for i in top_idx]

        # 全局缓冲
        for i in idx_c:
            global_buf.append((int(test_edges[i, 0]),
                               int(test_edges[i, 1]),
                               int(total_labels[i]),
                               float(total_pred[i, c])))

    global_top20 = sorted(global_buf, key=lambda x: x[3], reverse=True)[:topk_global]

    # 2. zero-shot
    seen = set(train_edges.flatten().tolist())
    zs_mask = [(int(d1) not in seen) or (int(d2) not in seen)
               for d1, d2 in test_edges]
    zs_pred = total_pred[zs_mask]
    zs_label = total_labels[zs_mask]

    # 3. 保存
    with open(os.path.join(out_dir, 'Deng_high_conf_top5_per_class.pkl'), 'wb') as f:
        pickle.dump(high_conf, f, protocol=4)

    with open(os.path.join(out_dir, 'Deng_high_conf_top20_global.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Drug1', 'Drug2', 'TrueLabel', 'Prob'])
        writer.writerows(global_top20)

    with open(os.path.join(out_dir, 'Deng_zero_shot_data.pkl'), 'wb') as f:
        pickle.dump({'zs_pred': zs_pred, 'zs_label': zs_label}, f, protocol=4)

    print(f'[INFO] 65-class case-study data saved -> {out_dir}')

#
# # ==================== 可视化工具函数 ====================
# # ---------- 注意力热图（每 50 epoch） ----------
# def vis_attention_heatmap_seq(model, train_edges, train_labels, device,
#                               epoch, sample_k=200, out_dir='vis/Deng/attn'):
#     os.makedirs(out_dir, exist_ok=True)
#     model.eval()
#     with torch.no_grad():
#         perm   = torch.randperm(len(train_edges))[:sample_k]
#         edges  = torch.from_numpy(train_edges[perm]).long().to(device)
#         labels = torch.from_numpy(train_labels[perm]).long()
#
#         edge_index, edge_type = get_edge_index(train_edges, train_labels)
#         diff_edge_index, _    = get_edge_index(new_train_edges, train_labels)
#         edge_index  = edge_index.to(device)
#         diff_edge_index = diff_edge_index.to(device)
#         edge_type   = edge_type.to(device)
#
#         ae_out, _   = model.ae(train_feat_mat)
#         d_g_out     = model.diff_gnn(ae_out, diff_edge_index)
#         o_g_out     = model.orgin_gnn(ae_out, edge_index, edge_type)
#         emb_stack   = torch.stack([ae_out, d_g_out, o_g_out], dim=1)
#         _, beta     = model.att_fusion(emb_stack)
#
#         heads, tails = edges[:, 0], edges[:, 1]
#         beta_pair    = (beta[heads] + beta[tails]) / 2
#         beta_np      = beta_pair.cpu().numpy()
#         labels_np    = labels.numpy()
#
#         # 独立保存
#         save_path = os.path.join(out_dir, f'attn_ep{epoch:03d}.pkl')
#         with open(save_path, 'wb') as f:
#             pickle.dump({'beta': beta_np, 'label': labels_np}, f)
#
#         # 绘制并保存 png（用于后续合成 GIF）
#         beta_np = beta_np.squeeze(-1)  # <-- 新增
#         df = pd.DataFrame(beta_np, columns=['AE', 'Diff-GNN', 'Origin-GNN'])
#         df['Label'] = labels_np
#         top10 = df['Label'].value_counts().index[:10]
#         beta_mean = df[df['Label'].isin(top10)].groupby('Label').mean()
#
#         plt.figure(figsize=(4, 6))
#         sns.heatmap(beta_mean, annot=True, fmt='.2f', cmap='Blues')
#         plt.title(f'Attention Heatmap  Epoch-{epoch}')
#         plt.tight_layout()
#         png_path = os.path.join(out_dir, f'attn_ep{epoch:03d}.png')
#         plt.savefig(png_path, dpi=120)
#         plt.close()
#         print(f'[ATTN] 已保存 {png_path}')
#
#
# # ---------- UMAP 嵌入对比（每 50 epoch） ----------
# def vis_umap_embed_seq(model, train_edges, train_labels, device,
#                        epoch, sample_k=2000, out_dir='vis/Deng/umap'):
#     os.makedirs(out_dir, exist_ok=True)
#     model.eval()
#     with torch.no_grad():
#         perm   = torch.randperm(len(train_edges))[:sample_k]
#         edges  = torch.from_numpy(train_edges[perm]).long().to(device)
#         labels = torch.from_numpy(train_labels[perm]).long()
#
#         edge_index, edge_type = get_edge_index(train_edges, train_labels)
#         diff_edge_index, _    = get_edge_index(new_train_edges, train_labels)
#         edge_index  = edge_index.to(device)
#         diff_edge_index = diff_edge_index.to(device)
#         edge_type   = edge_type.to(device)
#
#         ae_out, _ = model.ae(train_feat_mat)
#         d_g_out   = model.diff_gnn(ae_out, diff_edge_index)
#         o_g_out   = model.orgin_gnn(ae_out, edge_index, edge_type)
#         emb_stack = torch.stack([ae_out, d_g_out, o_g_out], dim=1)
#         emb, _    = model.att_fusion(emb_stack)
#
#         heads, tails = edges[:, 0], edges[:, 1]
#         pair_fus = (emb[heads] + emb[tails]) / 2
#
#         reducer = umap.UMAP(n_components=2, random_state=42)
#         emb_2d  = reducer.fit_transform(pair_fus.cpu().numpy())
#
#         # 独立保存
#         save_path = os.path.join(out_dir, f'umap_ep{epoch:03d}.pkl')
#         with open(save_path, 'wb') as f:
#             pickle.dump({'emb_2d': emb_2d, 'label': labels.numpy()}, f)
#
#         # 绘制并保存 png
#         plt.figure(figsize=(5, 4))
#         scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1],
#                               c=labels, s=8, cmap='tab10', alpha=0.8)
#         plt.colorbar(scatter, fraction=0.03)
#         plt.title(f'UMAP Epoch-{epoch}')
#         plt.tight_layout()
#         png_path = os.path.join(out_dir, f'umap_ep{epoch:03d}.png')
#         plt.savefig(png_path, dpi=120)
#         plt.close()
#         print(f'[UMAP] 已保存 {png_path}')
#
#
# # ========== 属性重要性：AUC + F1 双指标 ==========
# def vis_explain_epoch(model, edges, labels, device, epoch, attr_dims,
#                       attr_names=['Chem', 'Target', 'Enzyme', 'Pathway'],
#                       out_dir='vis/Deng/explain'):
#     os.makedirs(out_dir, exist_ok=True)
#     model.eval()
#     with torch.no_grad():
#         # ---- 0. 基准表现 ----
#         base_auc = quick_auc(model, edges, labels, device)
#         base_f1  = quick_f1(model, edges, labels, device)
#
#         auc_drops, f1_drops = [], []
#         start = 0
#         for name, dim in zip(attr_names, attr_dims):
#             end = start + dim
#             auc_drop, f1_drop = 0, 0
#             # 3 次平均减小随机波动
#             for _ in range(3):
#                 tmp_feat = train_feat_mat.clone()
#                 # 把该属性列随机洗牌
#                 tmp_feat[:, start:end] = tmp_feat[torch.randperm(tmp_feat.size(0)), start:end]
#                 auc_drop += (base_auc - quick_auc(model, edges, labels, device, tmp_feat))
#                 f1_drop  += (base_f1  - quick_f1(model, edges, labels, device, tmp_feat))
#             auc_drops.append(auc_drop / 3)
#             f1_drops.append(f1_drop / 3)
#             start = end
#
#         # ---- 1. 保存 CSV ----
#         csv_path = f'{out_dir}/attr_import_ep{epoch:03d}.csv'
#         pd.DataFrame({'Attribute': attr_names,
#                       'AUC_drop': auc_drops,
#                       'F1_drop':  f1_drops}).to_csv(csv_path, index=False, float_format='%.5f')
#
#         # ---- 2. 画 AUC 条形图 ----
#         order_auc = np.argsort(auc_drops)[::-1]
#         plt.figure(figsize=(5, 3))
#         plt.barh(np.array(attr_names)[order_auc], np.array(auc_drops)[order_auc],
#                  color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
#         plt.xlabel('AUC drop (Permutation)')
#         plt.title(f'Attribute Importance (AUC)  Epoch {epoch}')
#         plt.tight_layout()
#         for ext in ['png', 'pdf']:
#             plt.savefig(f'{out_dir}/attr_bar_auc_ep{epoch:03d}.{ext}', dpi=300)
#         plt.close()
#
#         # ---- 3. 画 F1 条形图 ----
#         order_f1 = np.argsort(f1_drops)[::-1]
#         plt.figure(figsize=(5, 3))
#         plt.barh(np.array(attr_names)[order_f1], np.array(f1_drops)[order_f1],
#                  color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
#         plt.xlabel('F1 drop (Permutation)')
#         plt.title(f'Attribute Importance (F1)  Epoch {epoch}')
#         plt.tight_layout()
#         for ext in ['png', 'pdf']:
#             plt.savefig(f'{out_dir}/attr_bar_f1_ep{epoch:03d}.{ext}', dpi=300)
#         plt.close()
#
#         print(f'[Explain] AUC & F1 importance figures + csv saved for epoch {epoch}')
#
#
# # ---- 辅助：快速计算 AUC ----
# def quick_auc(model, edges, labels, device, feat=None):
#     if feat is None:
#         feat = train_feat_mat
#     edge_index, edge_type = get_edge_index(train_edges, train_labels)
#     diff_edge_index, _      = get_edge_index(new_train_edges, train_labels)
#     edge_index  = edge_index.to(device)
#     diff_edge_index = diff_edge_index.to(device)
#     edge_type   = edge_type.to(device)
#
#     with torch.no_grad():
#         ae_out, _ = model.ae(feat)
#         d_g_out   = model.diff_gnn(ae_out, diff_edge_index)
#         o_g_out   = model.orgin_gnn(ae_out, edge_index, edge_type)
#         emb, _    = model.att_fusion(torch.stack([ae_out, d_g_out, o_g_out], dim=1))
#
#         h, t = torch.from_numpy(edges[:, 0]).long().to(device), \
#                torch.from_numpy(edges[:, 1]).long().to(device)
#         pair = (emb[h] * emb[t])
#         logits = model.predictor(pair)
#         prob = F.softmax(logits, dim=1).cpu().numpy()
#     y_true = label_binarize(labels, classes=range(65))
#     return roc_auc_score(y_true.ravel(), prob.ravel())
#
#
# # ---- 辅助：快速计算 macro-F1 ----
# def quick_f1(model, edges, labels, device, feat=None):
#     if feat is None:
#         feat = train_feat_mat
#     edge_index, edge_type = get_edge_index(train_edges, train_labels)
#     diff_edge_index, _      = get_edge_index(new_train_edges, train_labels)
#     edge_index  = edge_index.to(device)
#     diff_edge_index = diff_edge_index.to(device)
#     edge_type   = edge_type.to(device)
#
#     with torch.no_grad():
#         ae_out, _ = model.ae(feat)
#         d_g_out   = model.diff_gnn(ae_out, diff_edge_index)
#         o_g_out   = model.orgin_gnn(ae_out, edge_index, edge_type)
#         emb, _    = model.att_fusion(torch.stack([ae_out, d_g_out, o_g_out], dim=1))
#
#         h, t = torch.from_numpy(edges[:, 0]).long().to(device), \
#                torch.from_numpy(edges[:, 1]).long().to(device)
#         pair = (emb[h] * emb[t])
#         logits = model.predictor(pair)
#         pred = torch.argmax(logits, dim=1).cpu().numpy()
#     return f1_score(labels, pred, average='macro')
#
#
# # ========== 2D + 3D 聚类可视化 ==========
# def vis_cluster_epoch_2d3d(model, edges, labels, device, epoch,
#                              out_dir='vis/Deng/cluster'):
#     os.makedirs(out_dir, exist_ok=True)
#     model.eval()
#     with torch.no_grad():
#         # ---- 1. 提取 256 维边特征 ----
#         edge_index, edge_type = get_edge_index(train_edges, train_labels)
#         diff_edge_index, _      = get_edge_index(new_train_edges, train_labels)
#         edge_index  = edge_index.to(device)
#         diff_edge_index = diff_edge_index.to(device)
#         edge_type   = edge_type.to(device)
#
#         ae_out, _ = model.ae(train_feat_mat)
#         d_g_out   = model.diff_gnn(ae_out, diff_edge_index)
#         o_g_out   = model.orgin_gnn(ae_out, edge_index, edge_type)
#         emb, _    = model.att_fusion(torch.stack([ae_out, d_g_out, o_g_out], dim=1))
#
#         h, t = torch.from_numpy(edges[:, 0]).long(), \
#                torch.from_numpy(edges[:, 1]).long()
#         pair_feat = ((emb[h] + emb[t]) / 2).cpu().numpy()   # (M,256)
#
#         # ---- 2. PCA + UMAP ----
#         from sklearn.decomposition import PCA
#         from sklearn.metrics import normalized_mutual_info_score as NMI
#         from sklearn.metrics import adjusted_rand_score as ARI
#         import umap
#
#         pca50 = PCA(n_components=50, random_state=42)
#         low50 = pca50.fit_transform(pair_feat)
#
#         # 2D
#         umap2d = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
#                            metric='cosine', random_state=42)
#         emb2d = umap2d.fit_transform(low50)
#
#         # 3D
#         umap3d = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.3,
#                            metric='cosine', random_state=42)
#         emb3d = umap3d.fit_transform(low50)
#
#         # ---- 3. 聚类指标 ----
#         from sklearn.cluster import KMeans
#         k2 = KMeans(n_clusters=65, random_state=42).fit_predict(emb2d)
#         nmi2 = NMI(labels, k2)
#         ari2 = ARI(labels, k2)
#         print(f'[Cluster2D] epoch {epoch:03d}  NMI={nmi2:.4f}  ARI={ari2:.4f}')
#
#         # ---- 4. 2D 图 ----
#         plt.figure(figsize=(7, 5))
#         palette = sns.color_palette('tab20', 65)
#         sns.scatterplot(x=emb2d[:, 0], y=emb2d[:, 1],
#                         hue=labels, palette=palette,
#                         s=25, linewidth=0, legend=False)
#         plt.title(f'UMAP-2D  epoch {epoch}')
#         plt.tight_layout()
#         for ext in ['png', 'pdf']:
#             plt.savefig(f'{out_dir}/umap2d_ep{epoch:03d}.{ext}', dpi=300)
#         plt.close()
#
#         # ---- 5. 3D 图 ----
#         from mpl_toolkits.mplot3d import Axes3D
#         fig = plt.figure(figsize=(7, 5))
#         ax = fig.add_subplot(111, projection='3d')
#         # 颜色循环
#         colors = np.array([palette[i % 20] for i in labels])
#         ax.scatter(emb3d[:, 0], emb3d[:, 1], emb3d[:, 2],
#                    c=colors, s=25, linewidths=0)
#         ax.set_title(f'UMAP-3D  epoch {epoch}')
#         ax.dist = 10
#         for ext in ['png', 'pdf']:
#             plt.savefig(f'{out_dir}/umap3d_ep{epoch:03d}.{ext}', dpi=300)
#         plt.close()
#
#         # ---- 6. 存 3D 坐标与标签，方便后续转 GIF ----
#         np.save(f'{out_dir}/umap3d_ep{epoch:03d}.npy',
#                 {'xyz': emb3d, 'label': labels})
#
#         # ---- 7. 记录指标 ----
#         with open(f'{out_dir}/nmi_ari_2d.txt', 'a') as f:
#             f.write(f'{epoch}\t{nmi2:.4f}\t{ari2:.4f}\n')




def save_raw_edge_umap_fixed(idx, out_dir='vis/Deng/raw_edge'):
    os.makedirs(out_dir, exist_ok=True)
    edges = test_edges[idx]
    labels = test_labels[idx]

    feat = train_feat_mat.cpu().numpy()
    raw_pair_feat = np.hstack([feat[edges[:, 0]], feat[edges[:, 1]]])  # (N, 2*feat_dim)

    pca50 = PCA(n_components=50, random_state=42)
    low50 = pca50.fit_transform(raw_pair_feat)
    umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(low50)
    umap_3d = umap.UMAP(n_components=3, random_state=42).fit_transform(low50)

    df = pd.DataFrame({
        'edge_id': idx,
        'drug1': edges[:, 0],
        'drug2': edges[:, 1],
        'true_label': labels,
        'split': 'raw_edge',
        'umap_2d_1': umap_2d[:, 0],
        'umap_2d_2': umap_2d[:, 1],
        'umap_3d_1': umap_3d[:, 0],
        'umap_3d_2': umap_3d[:, 1],
        'umap_3d_3': umap_3d[:, 2],
    })
    csv_path = os.path.join(out_dir, 'raw_edge_umap_full.csv')
    df.to_csv(csv_path, index=False, float_format='%.5f')
    print(f'[RawEdgeUMAP] 全采样 + 固定索引已保存 -> {csv_path}')



def save_viz_csv_fixed(idx, model, edges, labels, device, epoch, out_dir='vis/Deng'):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        edge_index, edge_type = get_edge_index(train_edges, train_labels)
        diff_edge_index, _ = get_edge_index(new_train_edges, train_labels)
        edge_index  = edge_index.to(device)
        diff_edge_index = diff_edge_index.to(device)
        edge_type   = edge_type.to(device)

        ae_out, _ = model.ae(train_feat_mat)
        d_g_out   = model.diff_gnn(ae_out, diff_edge_index)
        o_g_out   = model.orgin_gnn(ae_out, edge_index, edge_type)
        emb, _    = model.att_fusion(torch.stack([ae_out, d_g_out, o_g_out], dim=1))

        h = torch.from_numpy(edges[:, 0]).long()
        t = torch.from_numpy(edges[:, 1]).long()
        pair_emb = ((emb[h] + emb[t]) / 2).detach().cpu().numpy()

        pca50 = PCA(n_components=50, random_state=42)
        low50 = pca50.fit_transform(pair_emb)
        umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(low50)
        umap_3d = umap.UMAP(n_components=3, random_state=42).fit_transform(low50)

        logits = model.predictor(torch.from_numpy(pair_emb).to(device))
        prob = F.softmax(logits, dim=1).detach().cpu().numpy()
        pred_l = prob.argmax(1)
        prob_max = prob.max(1)

        df = pd.DataFrame({
            'edge_id': idx,
            'drug1': edges[:, 0],
            'drug2': edges[:, 1],
            'true_label': labels,
            'pred_label': pred_l,
            'prob_max': prob_max,
            'embed_2d_u1': umap_2d[:, 0],
            'embed_2d_u2': umap_2d[:, 1],
            'embed_3d_u1': umap_3d[:, 0],
            'embed_3d_u2': umap_3d[:, 1],
            'embed_3d_u3': umap_3d[:, 2],
            'split': 'test'
        })
        csv_path = os.path.join(out_dir, f'viz_ep{epoch:03d}_full.csv')
        df.to_csv(csv_path, index=False, float_format='%.5f')
        print(f'[VizCSV] 全采样 + 固定索引已保存 -> {csv_path}')




train_()















#
# def save_viz_csv(model, edges, labels, device, epoch,
#                  out_dir='vis/Deng', sample_k=None):
#     """
#     把可视化刚需数据一次性写成 CSV
#     如果样本太多，用 sample_k 随机子采样（推荐 2k~5k 点，画图不卡）
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     model.eval()
#
#     # ---- 子采样（可选） ----
#     if sample_k is not None and len(edges) > sample_k:
#         idx = np.random.choice(len(edges), sample_k, replace=False)
#         edges  = edges[idx]
#         labels = labels[idx]
#
#     # ---- 提取边嵌入（256） ----
#     edge_index, edge_type = get_edge_index(train_edges, train_labels)
#     diff_edge_index, _    = get_edge_index(new_train_edges, train_labels)
#     edge_index  = edge_index.to(device)
#     diff_edge_index = diff_edge_index.to(device)
#     edge_type   = edge_type.to(device)
#
#     ae_out, _ = model.ae(train_feat_mat)
#     d_g_out   = model.diff_gnn(ae_out, diff_edge_index)
#     o_g_out   = model.orgin_gnn(ae_out, edge_index, edge_type)
#     emb, _    = model.att_fusion(torch.stack([ae_out, d_g_out, o_g_out], dim=1))
#
#     h, t = torch.from_numpy(edges[:, 0]).long(), \
#            torch.from_numpy(edges[:, 1]).long()
#     pair_emb = ((emb[h] + emb[t]) / 2).detach().cpu().numpy()  # M×256
#
#     # ---- PCA→UMAP 2D/3D ----
#     pca50 = PCA(n_components=50, random_state=42)
#     low50 = pca50.fit_transform(pair_emb)
#
#     umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(low50)
#     umap_3d = umap.UMAP(n_components=3, random_state=42).fit_transform(low50)
#
#     # ---- 预测结果 ----
#     logits = model.predictor(torch.from_numpy(pair_emb).to(device))
#     prob = F.softmax(logits, dim=1).detach().cpu().numpy()
#     pred_l = prob.argmax(1)
#     prob_max = prob.max(1)
#
#     # ---- 组装 DataFrame ----
#     df = pd.DataFrame({
#         'edge_id'     : np.arange(len(edges)),
#         'drug1'       : edges[:, 0],
#         'drug2'       : edges[:, 1],
#         'true_label'  : labels,
#         'pred_label'  : pred_l,
#         'prob_max'    : prob_max,
#         'embed_2d_u1' : umap_2d[:, 0],
#         'embed_2d_u2' : umap_2d[:, 1],
#         'embed_3d_u1' : umap_3d[:, 0],
#         'embed_3d_u2' : umap_3d[:, 1],
#         'embed_3d_u3' : umap_3d[:, 2],
#         'split'       : 'test'
#     })
#
#     # ---- 把 65 维概率展开（可选） ----
#     prob_cols = {f'prob_{i}': prob[:, i] for i in range(prob.shape[1])}
#     df = pd.concat([df, pd.DataFrame(prob_cols)], axis=1)
#
#     csv_path = os.path.join(out_dir, f'viz_ep{epoch:03d}.csv')
#     df.to_csv(csv_path, index=False, float_format='%.5f')
#     print(f'[VizCSV] 已保存 {csv_path}  ({len(df)} 条边)')
#
#
# def save_raw_edge_umap(out_dir='vis/Deng/raw_edge', sample_k=None):
#     """
#     对测试集边的原始特征（未训练）做 UMAP 2D/3D 降维
#     用于和训练后的嵌入空间对比
#     """
#     os.makedirs(out_dir, exist_ok=True)
#
#     # ---- 采样测试边（可选） ----
#     edges = test_edges
#     labels = test_labels
#     if sample_k is not None and len(edges) > sample_k:
#         idx = np.random.choice(len(edges), sample_k, replace=False)
#         edges = edges[idx]
#         labels = labels[idx]
#
#     # ---- 构造原始边特征：拼接 head & tail 的原始特征 ----
#     feat = train_feat_mat.cpu().numpy()
#     h = edges[:, 0]
#     t = edges[:, 1]
#     raw_pair_feat = np.hstack([feat[h], feat[t]])  # (N, 2*feat_dim)
#
#     # ---- PCA → UMAP ----
#     pca50 = PCA(n_components=50, random_state=42)
#     low50 = pca50.fit_transform(raw_pair_feat)
#
#     umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(low50)
#     umap_3d = umap.UMAP(n_components=3, random_state=42).fit_transform(low50)
#
#     # ---- 保存 CSV ----
#     df = pd.DataFrame({
#         'edge_id': np.arange(len(edges)),
#         'drug1': edges[:, 0],
#         'drug2': edges[:, 1],
#         'true_label': labels,
#         'split': 'raw_edge',
#         'umap_2d_1': umap_2d[:, 0],
#         'umap_2d_2': umap_2d[:, 1],
#         'umap_3d_1': umap_3d[:, 0],
#         'umap_3d_2': umap_3d[:, 1],
#         'umap_3d_3': umap_3d[:, 2],
#     })
#     csv_path = os.path.join(out_dir, 'raw_edge_umap.csv')
#     df.to_csv(csv_path, index=False, float_format='%.5f')
#     print(f'[RawEdgeUMAP] 测试集原始边特征 UMAP 已保存 -> {csv_path}')
#
# save_raw_edge_umap(sample_k=None)  # 全采样
