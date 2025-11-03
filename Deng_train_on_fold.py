import torch
import argparse
import os
import sys
import time
from collections import defaultdict
from functools import reduce

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset, DataLoader

from Deng_model import Model, get_edge_index
from utils import get_adj_mat, load_feat
#get_edge_index生成图的边索引和边类型张量
#get_adj_mat，读取边数据，构建一个无向图，并提取图的边列表和节点总数
##从指定的文件中加载特征数据，并将这些数据转换为 PyTorch 张量
'''Code for 5 fold cross-validation'''

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_dim', type=int, default=512, help='num of hidden features')
parser.add_argument('--out_dim', type=int, default=256, help='num of output features')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout rate')
parser.add_argument('--rels', type=int, default=65, help='num of relations')
parser.add_argument('--n_epochs', type=int, default=400, help='num of epochs')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--threshold', type=float, default=0.4, help='edge threshold')
args = parser.parse_args()


hidden_dim = args.hidden_dim
out_dim = args.out_dim
n_epochs = args.n_epochs
batch_size = args.batch_size

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# if torch.cuda.device_count() > 1:
#     device = torch.device("cuda:1")  # 使用第二张卡，索引为1
# else:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将一个 SciPy 稀疏矩阵转换为 PyTorch 的稀疏张量
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


edges_all, num_nodes = get_adj_mat()  # Get all DDIs in dataset


# 数据读取与分层抽样将每个类别的数据按比例分割为测试集和训练集，并将结果存储在 splits 列表中。
def get_straitified_data(ratio=0.2):
    df_data = pd.read_csv('Deng/ddi_class_65.csv')
    all_tup = [(h, t, r) for h, t, r in zip(df_data['Drug1'], df_data['Drug2'], df_data['Label'])]
    np.random.shuffle(all_tup)
    tuple_by_type = defaultdict(list)
    for h, t, r in all_tup:
        tuple_by_type[r].append((h, t, r))
    tuple_by_type.keys().__len__()
    train_edges = []
    test_edges = []
    splits = []

    for k in range(0, (int)(1 / ratio)):
        edges = []
        for r in tuple_by_type.keys():
            test_set_size = int(len(tuple_by_type[r]) * ratio)
            if k < 4:
                edges.append(tuple_by_type[r][k * test_set_size:(k + 1) * test_set_size])
            else:
                edges.append(tuple_by_type[r][k * test_set_size:])
        splits.append(edges)
    return splits


# Generate training and test set for each fold
def make_data(splits, fold_k):
    test_edges = splits[fold_k]
    train_edges = []
    for i in range(0, len(splits)):
        if i == fold_k:
            continue
        train_edges += splits[i]

    def merge(x, y):
        return x + y

    test_tups = np.array(reduce(merge, test_edges))

    train_tups = np.array(reduce(merge, train_edges))

    train_edges = train_tups[:, :2]
    train_labels = train_tups[:, -1]
    test_edges = test_tups[:, :2]
    test_labels = test_tups[:, -1]
    return train_edges, train_labels, test_edges, test_labels


testset_ratio = 0.2
splits = get_straitified_data(testset_ratio)

feat_mat_chem, sim_mat_chem = load_feat('Deng/chem_Jacarrd_sim.csv')  # Load drug attribute file
feat_mat_target, sim_mat_target = load_feat('Deng/target_Jacarrd_sim.csv')  # Load drug attribute file
feat_mat_enzyme, sim_mat_enzyme = load_feat('Deng/enzyme_Jacarrd_sim.csv')  # Load drug attribute file
feat_mat_pathway, sim_mat_pathway = load_feat('Deng/pathway_Jacarrd_sim.csv')  # Load drug attribute file
train_feat_mat = torch.hstack([feat_mat_chem, feat_mat_target, feat_mat_enzyme, feat_mat_pathway])
train_feat_mat = torch.FloatTensor(train_feat_mat)
train_feat_mat = train_feat_mat.to(device)
args.feat_dim = train_feat_mat.shape[1]


#根据相似度矩阵生成新的边
def generate_edges_from_similarity(train_edges, similarity_matrix, threshold):
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
    train_nodes = np.unique(train_edges.flatten())
    num_nodes = similarity_matrix.shape[0]
    new_edges = []

    # 将训练集节点转换为集合，方便快速查找
    train_nodes_set = set(train_nodes)

    for i in range(num_nodes):
        # 仅处理训练集中的节点
        if i not in train_nodes_set:
            continue
        for j in range(i + 1, num_nodes):
            # 仅处理训练集中的节点
            if j not in train_nodes_set:
                continue
            if similarity_matrix[i, j] > threshold:
                new_edges.append([i, j])

    return np.array(new_edges)




print(os.path.basename(sys.argv[0]))
print("training set ratio:", (1 - testset_ratio))


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


results = []  # Record the prediction results for each fold

'''Perform 5 fold cross-validation'''
for k in range(0, 5):

    train_edges, train_labels, test_edges, test_labels = make_data(splits, k)
    train_dataset = DDIDataset(train_edges, train_labels)
    test_dataset = DDIDataset(test_edges, test_labels)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    new_edges_chem = generate_edges_from_similarity(train_edges, sim_mat_chem.cpu().numpy(), threshold=args.threshold)
    new_edges_target = generate_edges_from_similarity(train_edges, sim_mat_target.cpu().numpy(),
                                                      threshold=args.threshold)
    new_edges_enzyme = generate_edges_from_similarity(train_edges, sim_mat_enzyme.cpu().numpy(),
                                                      threshold=args.threshold)
    new_edges_pathway = generate_edges_from_similarity(train_edges, sim_mat_pathway.cpu().numpy(),
                                                       threshold=args.threshold)
    # 合并所有新的边
    new_train_edges = np.vstack([train_edges, new_edges_chem, new_edges_target, new_edges_enzyme, new_edges_pathway])
    # 去除重复的边
    new_train_edges = np.unique(new_train_edges, axis=0)

    n_iterations = len(train_loader)
    num_epochs = n_epochs
    start = time.time()

    running_loss = 0.0
    running_correct = 0.0

    model = Model(args)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    re_criterion = nn.MSELoss().to(device)
    trainer_E = torch.optim.Adam(model.parameters(), lr=1e-3)

    # num_epochs = 1
    start = time.time()

    running_loss = 0.0
    running_correct = 0.0
    # Training phase
    for epoch in range(num_epochs):
        model.train()
        true_labels, pred_labels = [], []
        running_loss = 0.0
        running_correct = 0.0
        total_samples = 0
        for i, (edges, labels) in enumerate(train_loader):
            edges, labels = edges.to(device), labels.to(device)
            edge_index, edge_type = get_edge_index(train_edges, train_labels)
            diff_edge_index, _ = get_edge_index(new_train_edges, train_labels)
            edge_index, edge_type, diff_edge_index = edge_index.to(device), edge_type.to(device), diff_edge_index.to(
                device)
            y_pred, re_out = model(edge_index, diff_edge_index, edge_type, train_feat_mat, edges)
            loss = criterion(y_pred, labels)
            re_loss = re_criterion(re_out, train_feat_mat)
            loss += re_loss

            trainer_E.zero_grad()
            loss.backward()
            trainer_E.step()

            running_correct += torch.sum(
                (torch.argmax(y_pred, dim=1).type(torch.FloatTensor) == labels.cpu()).detach()).float()
            pred_labels.append(list(y_pred.cpu().detach().numpy().reshape(-1)))

            labels = labels.cpu().numpy()
            total_samples += labels.shape[0]
            true_labels.append(list(labels))
            running_loss += loss.item()

        print(
            f"fold {k + 1}-epoch {epoch + 1}/{num_epochs};trainging loss: {running_loss / n_iterations:.4f};training set acc: {running_correct / total_samples:.4f}")


        def merge(x, y):
            return x + y

    end = time.time()
    elapsed = end - start
    print(f"Training completed in {elapsed // 60}m: {elapsed % 60:.2f}s.")


    def roc_aupr_score(y_true, y_score, average="macro"):
        def _binary_roc_aupr_score(y_true, y_score):
            precision, recall, _ = precision_recall_curve(y_true, y_score)

            return auc(recall, precision)

        def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
            if average == "binary":
                return binary_metric(y_true, y_score)
            if average == "micro":
                y_true = y_true.ravel()
                y_score = y_score.ravel()
            if y_true.ndim == 1:
                y_true = y_true.reshape((-1, 1))
            if y_score.ndim == 1:
                y_score = y_score.reshape((-1, 1))
            n_classes = y_score.shape[1]
            score = np.zeros((n_classes,))
            for c in range(n_classes):
                y_true_c = y_true.take([c], axis=1).ravel()
                y_score_c = y_score.take([c], axis=1).ravel()
                score[c] = binary_metric(y_true_c, y_score_c)
            return np.average(score)

        return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)


    event_num = 65
    n_test_samples = 0
    n_correct = 0
    total_labels = []
    total_pred = []
    fold_results = []

    # Testing phase
    with torch.no_grad():
        model.eval()
        for edges, labels in test_loader:
            edge_index, edge_type = get_edge_index(train_edges, train_labels)
            diff_edge_index, _ = get_edge_index(new_train_edges, train_labels)
            edge_index, edge_type, diff_edge_index = edge_index.to(device), edge_type.to(device), diff_edge_index.to(
                device)
            edges, labels = edges.to(device), labels.to(device)
            y_pred, _ = model(edge_index, diff_edge_index, edge_type, train_feat_mat, edges, False)
            y_hat = F.softmax(y_pred, dim=1)
            total_pred.append(y_hat.cpu().numpy())

            total_labels.append(labels.cpu().numpy())

            n_test_samples += edges.shape[0]

            n_correct += torch.sum(
                (torch.argmax(y_pred, dim=1).type(torch.FloatTensor) == labels.cpu()).detach()).float()

        acc = 100.0 * n_correct / n_test_samples

        total_pred = np.vstack(total_pred)
        total_labels = np.concatenate(total_labels)
        pred_type = np.argmax(total_pred, axis=1)
        y_one_hot = label_binarize(total_labels, classes=np.arange(event_num))

        aupr = roc_aupr_score(y_one_hot, total_pred, average='micro')
        auroc = roc_auc_score(y_one_hot, total_pred, average='micro')
        f1 = f1_score(total_labels, pred_type, average='macro')
        precision = precision_score(total_labels, pred_type, average='macro', zero_division=0)
        recall = recall_score(total_labels, pred_type, average='macro', zero_division=0)
        f1_mi = f1_score(total_labels, pred_type, average='micro')
        precision_mi = precision_score(total_labels, pred_type, average='micro', zero_division=0)
        recall_mi = recall_score(total_labels, pred_type, average='micro', zero_division=0)

        # # 计算每个类别的 AUCPR 和 F1 分数
        # aucpr_scores = []
        # f1_scores = []
        # for i in range(event_num):
        #     precision_class, recall_class, _ = precision_recall_curve(y_one_hot[:, i], total_pred[:, i])
        #     aucpr_scores.append(auc(recall_class, precision_class))
        #     f1_scores.append(f1_score(y_one_hot[:, i], (total_pred[:, i] > 0.5).astype(int), zero_division=0))
        #
        # # 保存每个类别的 AUCPR 和 F1 分数到新文件
        # class_results = []
        # for i in range(event_num):
        #     class_results.append([i, aucpr_scores[i], f1_scores[i]])
        # class_results_df = pd.DataFrame(class_results, columns=['Class', 'AUCPR', 'F1'])
        # class_results_df.to_csv(f'1Deng_class_metrics_fold_{k}.csv', index=False)

        # 计算每个类别的 AUCPR、F1、Precision、Recall
        aucpr_scores = []
        f1_scores    = []
        prec_scores  = []
        rec_scores   = []
        for i in range(event_num):
            # AUCPR
            precision_class, recall_class, _ = precision_recall_curve(y_one_hot[:, i], total_pred[:, i])
            aucpr_scores.append(auc(recall_class, precision_class))
            # 二分类预测（0.5 阈值）
            y_true_i = y_one_hot[:, i]
            y_pred_i = (total_pred[:, i] > 0.5).astype(int)
            f1_scores.append(f1_score(y_true_i, y_pred_i, zero_division=0))
            prec_scores.append(precision_score(y_true_i, y_pred_i, zero_division=0))
            rec_scores.append(recall_score(y_true_i, y_pred_i, zero_division=0))

        # 保存到 DataFrame
        class_results = []
        for i in range(event_num):
            class_results.append([i, aucpr_scores[i], f1_scores[i], prec_scores[i], rec_scores[i]])
        class_results_df = pd.DataFrame(class_results,
                                        columns=['Class', 'AUCPR', 'F1', 'Precision', 'Recall'])
        class_results_df.to_csv(f'1Deng_class_metrics_fold_{k}.csv', index=False)


        fold_results.append(k)
        fold_results.append(acc)
        fold_results.append(aupr)
        fold_results.append(auroc)
        fold_results.append(f1)
        fold_results.append(precision)
        fold_results.append(recall)
        fold_results.append(f1_mi)
        fold_results.append(precision_mi)
        fold_results.append(recall_mi)
        print(f"test set accuracy: {acc}")
        print(f"AUPR: {aupr}")
        print(f"AUROC: {auroc}")
        print(f"F1: {f1}")
        print(f"Precison: {precision}")
        print(f"Recall: {recall}")
        print(f"F1_micro: {f1_mi}")
        print(f"Precison_micro: {precision_mi}")
        print(f"Recall_micro: {recall_mi}")

    results.append(fold_results)

# Write cross-validation results to file
pd.DataFrame(results,
             columns=['fold', 'acc', 'aupr', 'auroc', 'f1', 'precision', 'recall', 'f1_micro', 'precision_micro',
                      'recall_micro']).to_csv('1Deng_cross_validation.csv')
