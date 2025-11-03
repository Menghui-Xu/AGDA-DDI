import numpy as np
from torch_geometric.nn import RGCNConv, SAGEConv, GATConv, GCNConv,RGATConv

'''Model definition'''
import torch.nn.functional as F #，提供了各种神经网络层和函数的实现，如激活函数、池化函数等
import torch
import torch.nn as nn #提供了构建神经网络所需的各种层和模块，如全连接层、卷积层、循环层等。


class AutoEncoder(nn.Module):
    # input_dim：输入数据的维度。hidden_dim：隐藏层的维度。latent_dim：潜在空间的维度（编码后的表示维度）
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AutoEncoder, self).__init__()

        # 编码器
        # 将输入维度映射到两倍的隐藏维度。
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ELU(),                                    #  指数线性单元（Exponential Linear Unit）激活函数
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.2),
        )

        # 将两倍的隐藏维度映射到隐藏维度。
        self.encoder2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
        )

        # 将隐藏维度映射到潜在维度。
        self.encoder3 = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ELU(),
            nn.BatchNorm1d(latent_dim),
        )

        # 解码器
        #解码器将潜在维度映射回原始输入维度
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Linear(hidden_dim * 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z1 = self.encoder1(x)
        z2 = self.encoder2(z1)
        z3 = self.encoder3(z2)
        reconstructed = self.decoder(z3)  #解码器的输出，即重构的输入数据
        return torch.hstack([z1, z2, z3]), reconstructed #返回编码器各层的输出和重构的输入数据。


class SimpleGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(SimpleGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels * 2)
        self.conv2 = SAGEConv(hidden_channels * 2, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        # self.conv1 = GATConv(in_channels, hidden_channels * 2, heads=8, concat=False)
        # self.conv2 = GATConv(hidden_channels * 2, hidden_channels, heads=8, concat=False)
        # self.conv3 = GATConv(hidden_channels, out_channels, heads=8, concat=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h3 = self.conv3(h2, edge_index)
        h3 = F.relu(h3)
        return torch.hstack([h1, h2, h3])


class SimpleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(SimpleGAT, self).__init__()
        # self.conv1 = SAGEConv(in_channels, hidden_channels * 2)
        # self.conv2 = SAGEConv(hidden_channels * 2, hidden_channels)
        # self.conv3 = SAGEConv(hidden_channels, out_channels)
        # 多头注意力的输出取平均而非拼接，保持维度不变
        self.conv1 = GATConv(in_channels, hidden_channels * 2, heads=8, concat=False)
        self.conv2 = GATConv(hidden_channels * 2, hidden_channels, heads=8, concat=False)
        self.conv3 = GATConv(hidden_channels, out_channels, heads=8, concat=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h3 = self.conv3(h2, edge_index)
        h3 = F.relu(h3)
        return torch.hstack([h1, h2, h3])


class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(SimpleGCN, self).__init__()
        # self.conv1 = SAGEConv(in_channels, hidden_channels * 2)
        # self.conv2 = SAGEConv(hidden_channels * 2, hidden_channels)
        # self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.conv1 = GCNConv(in_channels, hidden_channels * 2)
        self.conv2 = GCNConv(hidden_channels * 2, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h3 = self.conv3(h2, edge_index)
        h3 = F.relu(h3)
        return torch.hstack([h1, h2, h3])


class R_GCN(torch.nn.Module):
    def __init__(self, num_feat, num_outputs, num_hidden, dropout, rels):
        super(R_GCN, self).__init__()
        # Define two conv layers

        self.conv1 = RGCNConv(num_feat, num_hidden * 2, num_relations=rels)
        self.conv2 = RGCNConv(num_hidden * 2, num_hidden, num_relations=rels)
        self.conv3 = RGCNConv(num_hidden, num_outputs, num_relations=rels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        h1 = self.conv1(x, edge_index, edge_type)
        h1 = F.elu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.conv2(h1, edge_index, edge_type)
        h2 = F.elu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h3 = self.conv3(h2, edge_index, edge_type)
        h3 = F.elu(h3)
        return torch.hstack([h1, h2, h3])


#动态计算输入特征的权重并生成加权聚合输出
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=8):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),              # # 非线性激活
            nn.Linear(hidden_size, 1, bias=False)  ## 生成注意力分数
        )

    def forward(self, z):
        w = self.project(z)                  ## 计算原始注意力分数 [N, 1]
        beta = torch.softmax(w, dim=1)       ## 归一化为概率分布 [N, 1]
        return (beta * z).sum(1), beta       # # 加权和 [D], 注意力权重 [N, 1]



# 深度全连接神经网络。数据依次通过各层（线性变换 → 激活 → 归一化 → Dropout）
class DNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 512),
            nn.ELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        output = self.layers(x)
        return output



#使用 SimpleGraphSAGE 作为图神经网络层
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.ae = AutoEncoder(input_dim=args.feat_dim, latent_dim=args.out_dim, hidden_dim=args.hidden_dim)
        self.diff_gnn = SimpleGraphSAGE(in_channels=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                                        hidden_channels=args.hidden_dim,
                                        out_channels=args.out_dim, dropout=args.dropout)
        self.orgin_gnn = R_GCN(num_feat=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                               num_hidden=args.hidden_dim,
                               num_outputs=args.out_dim, dropout=args.dropout,
                               rels=args.rels)  # Topology structure encoder
        self.att_fusion = Attention(args.hidden_dim * 2 + args.hidden_dim + args.out_dim)
        self.predictor = DNN((args.hidden_dim * 2 + args.hidden_dim + args.out_dim), args.rels)

    def forward(self, edge_index, diff_edge_index, edge_type, attr_mtx, triples, *args):
        ae_out, re_out = self.ae(attr_mtx)

        d_g_out = self.diff_gnn(ae_out, diff_edge_index)
        o_g_out = self.orgin_gnn(ae_out, edge_index, edge_type)


        emb = torch.stack([ae_out, d_g_out, o_g_out], dim=1)
        emb, _ = self.att_fusion(emb)


        triples = triples.long()

        heads = triples[:, 0]
        tails = triples[:, 1]


        # drug_pair = torch.hstack([emb[heads], emb[tails]])
        drug_pair = (emb[heads] + emb[tails]) / 2
        output = self.predictor(drug_pair)
        return output, re_out
        # # ==================== 核心修改：处理不同形状的 triples ====================
        # if triples.dim() == 2 and triples.size(1) == 2:
        #     # 正常训练时，triples 是 (batch_size, 2)
        #     heads = triples[:, 0]
        #     tails = triples[:, 1]
        #     # drug_pair = torch.hstack([emb[heads], emb[tails]])
        #     drug_pair = (emb[heads] + emb[tails]) / 2
        #     output = self.predictor(drug_pair)
        #     return output, re_out
        # elif triples.dim() == 2 and triples.size(1) == 1:
        #     # 可视化时，triples 是 (num_nodes, 1)
        #     # 此时我们只需要返回所有节点的嵌入，不需要经过 predictor
        #     return emb, re_out
        # else:
        #     raise ValueError(f"Unexpected triples shape: {triples.shape}")
        # # ========================================================================



#使用 SimpleGAT（图注意力网络）作为图神经网络层
class Model_GAT(nn.Module):
    def __init__(self, args):
        super(Model_GAT, self).__init__()
        self.ae = AutoEncoder(input_dim=args.feat_dim, latent_dim=args.out_dim, hidden_dim=args.hidden_dim)
        self.diff_gnn = SimpleGAT(in_channels=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                                  hidden_channels=args.hidden_dim,
                                  out_channels=args.out_dim, dropout=args.dropout)
        self.orgin_gnn = R_GCN(num_feat=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                               num_hidden=args.hidden_dim,
                               num_outputs=args.out_dim, dropout=args.dropout,
                               rels=args.rels)  # Topology structure encoder
        self.att_fusion = Attention(args.hidden_dim * 2 + args.hidden_dim + args.out_dim)
        self.predictor = DNN((args.hidden_dim * 2 + args.hidden_dim + args.out_dim), args.rels)

    def forward(self, edge_index, diff_edge_index, edge_type, attr_mtx, triples, *args):
        ae_out, re_out = self.ae(attr_mtx)

        d_g_out = self.diff_gnn(ae_out, diff_edge_index)
        o_g_out = self.orgin_gnn(ae_out, edge_index, edge_type)

        emb = torch.stack([ae_out, d_g_out, o_g_out], dim=1)
        emb, _ = self.att_fusion(emb)

        triples = triples.long()
        heads = triples[:, 0]
        tails = triples[:, 1]

        # drug_pair = torch.hstack([emb[heads], emb[tails]])
        drug_pair = (emb[heads] + emb[tails]) / 2
        output = self.predictor(drug_pair)
        return output, re_out


#使用 SimpleGCN（图卷积网络）作为图神经网络层
class Model_GCN(nn.Module):
    def __init__(self, args):
        super(Model_GCN, self).__init__()
        self.ae = AutoEncoder(input_dim=args.feat_dim, latent_dim=args.out_dim, hidden_dim=args.hidden_dim)
        self.diff_gnn = SimpleGCN(in_channels=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                                  hidden_channels=args.hidden_dim,
                                  out_channels=args.out_dim, dropout=args.dropout)
        self.orgin_gnn = R_GCN(num_feat=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                               num_hidden=args.hidden_dim,
                               num_outputs=args.out_dim, dropout=args.dropout,
                               rels=args.rels)  # Topology structure encoder
        self.att_fusion = Attention(args.hidden_dim * 2 + args.hidden_dim + args.out_dim)
        self.predictor = DNN((args.hidden_dim * 2 + args.hidden_dim + args.out_dim), args.rels)

    def forward(self, edge_index, diff_edge_index, edge_type, attr_mtx, triples, *args):
        ae_out, re_out = self.ae(attr_mtx)

        d_g_out = self.diff_gnn(ae_out, diff_edge_index)
        o_g_out = self.orgin_gnn(ae_out, edge_index, edge_type)

        emb = torch.stack([ae_out, d_g_out, o_g_out], dim=1)
        emb, _ = self.att_fusion(emb)

        triples = triples.long()
        heads = triples[:, 0]
        tails = triples[:, 1]

        # drug_pair = torch.hstack([emb[heads], emb[tails]])
        drug_pair = (emb[heads] + emb[tails]) / 2
        output = self.predictor(drug_pair)
        return output, re_out



#在融合头节点和尾节点的嵌入表示时，使用求和操作代替平均操作
class Model_sum(nn.Module):
    def __init__(self, args):
        super(Model_sum, self).__init__()
        self.ae = AutoEncoder(input_dim=args.feat_dim, latent_dim=args.out_dim, hidden_dim=args.hidden_dim)
        self.diff_gnn = SimpleGraphSAGE(in_channels=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                                        hidden_channels=args.hidden_dim,
                                        out_channels=args.out_dim, dropout=args.dropout)
        self.orgin_gnn = R_GCN(num_feat=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                               num_hidden=args.hidden_dim,
                               num_outputs=args.out_dim, dropout=args.dropout,
                               rels=args.rels)  # Topology structure encoder
        self.att_fusion = Attention(args.hidden_dim * 2 + args.hidden_dim + args.out_dim)
        self.predictor = DNN((args.hidden_dim * 2 + args.hidden_dim + args.out_dim), args.rels)

    def forward(self, edge_index, diff_edge_index, edge_type, attr_mtx, triples, *args):
        ae_out, re_out = self.ae(attr_mtx)

        d_g_out = self.diff_gnn(ae_out, diff_edge_index)
        o_g_out = self.orgin_gnn(ae_out, edge_index, edge_type)

        emb = torch.stack([ae_out, d_g_out, o_g_out], dim=1)
        emb, _ = self.att_fusion(emb)

        triples = triples.long()
        heads = triples[:, 0]
        tails = triples[:, 1]

        # drug_pair = torch.hstack([emb[heads], emb[tails]])
        # drug_pair = torch.mul(emb[heads],emb[tails])
        drug_pair = emb[heads] + emb[tails]
        output = self.predictor(drug_pair)
        return output, re_out


#使用拼接值融合头节点和尾节点的嵌入表示
class Model_cat(nn.Module):
    def __init__(self, args):
        super(Model_cat, self).__init__()
        self.ae = AutoEncoder(input_dim=args.feat_dim, latent_dim=args.out_dim, hidden_dim=args.hidden_dim)
        self.diff_gnn = SimpleGraphSAGE(in_channels=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                                        hidden_channels=args.hidden_dim,
                                        out_channels=args.out_dim, dropout=args.dropout)
        self.orgin_gnn = R_GCN(num_feat=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                               num_hidden=args.hidden_dim,
                               num_outputs=args.out_dim, dropout=args.dropout,
                               rels=args.rels)  # Topology structure encoder
        self.att_fusion = Attention(args.hidden_dim * 2 + args.hidden_dim + args.out_dim)
        self.predictor = DNN( 2*(args.hidden_dim * 2 + args.hidden_dim + args.out_dim), args.rels)

    def forward(self, edge_index, diff_edge_index, edge_type, attr_mtx, triples, *args):
        ae_out, re_out = self.ae(attr_mtx)

        d_g_out = self.diff_gnn(ae_out, diff_edge_index)
        o_g_out = self.orgin_gnn(ae_out, edge_index, edge_type)

        emb = torch.stack([ae_out, d_g_out, o_g_out], dim=1)
        emb, _ = self.att_fusion(emb)

        triples = triples.long()
        heads = triples[:, 0]
        tails = triples[:, 1]

        drug_pair = torch.hstack([emb[heads], emb[tails]])
        # drug_pair = torch.mul(emb[heads],emb[tails])
        # drug_pair = (emb[heads] + emb[tails]) / 2
        # drug_pair = torch.cat([emb[heads], emb[tails]], dim=1)
        output = self.predictor(drug_pair)
        return output, re_out


#使用平均值融合头节点和尾节点的嵌入表示
class Model_avg(nn.Module):
    def __init__(self, args):
        super(Model_avg, self).__init__()
        self.ae = AutoEncoder(input_dim=args.feat_dim, latent_dim=args.out_dim, hidden_dim=args.hidden_dim)
        self.diff_gnn = SimpleGraphSAGE(in_channels=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                                        hidden_channels=args.hidden_dim,
                                        out_channels=args.out_dim, dropout=args.dropout)
        self.orgin_gnn = R_GCN(num_feat=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                               num_hidden=args.hidden_dim,
                               num_outputs=args.out_dim, dropout=args.dropout,
                               rels=args.rels)  # Topology structure encoder
        self.att_fusion = Attention(args.hidden_dim * 2 + args.hidden_dim + args.out_dim)
        self.predictor = DNN((args.hidden_dim * 2 + args.hidden_dim + args.out_dim), args.rels)

    def forward(self, edge_index, diff_edge_index, edge_type, attr_mtx, triples, *args):
        ae_out, re_out = self.ae(attr_mtx)

        d_g_out = self.diff_gnn(ae_out, diff_edge_index)
        o_g_out = self.orgin_gnn(ae_out, edge_index, edge_type)

        emb = torch.stack([ae_out, d_g_out, o_g_out], dim=1)
        emb, _ = self.att_fusion(emb)

        triples = triples.long()
        heads = triples[:, 0]
        tails = triples[:, 1]

        # drug_pair = torch.hstack([emb[heads], emb[tails]])  水平拼接
        # drug_pair = torch.mul(emb[heads],emb[tails])    逐元素相乘
        drug_pair = (emb[heads] + emb[tails]) / 2
        output = self.predictor(drug_pair)
        return output, re_out



#使用哈达玛积融合头节点和尾节点的嵌入表示
class Model_hdm(nn.Module):
    def __init__(self, args):
        super(Model_hdm, self).__init__()
        self.ae = AutoEncoder(input_dim=args.feat_dim, latent_dim=args.out_dim, hidden_dim=args.hidden_dim)
        self.diff_gnn = SimpleGraphSAGE(in_channels=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                                        hidden_channels=args.hidden_dim,
                                        out_channels=args.out_dim, dropout=args.dropout)
        self.orgin_gnn = R_GCN(num_feat=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                               num_hidden=args.hidden_dim,
                               num_outputs=args.out_dim, dropout=args.dropout,
                               rels=args.rels)  # Topology structure encoder
        self.att_fusion = Attention(args.hidden_dim * 2 + args.hidden_dim + args.out_dim)
        self.predictor = DNN((args.hidden_dim * 2 + args.hidden_dim + args.out_dim), args.rels)

    def forward(self, edge_index, diff_edge_index, edge_type, attr_mtx, triples, *args):
        ae_out, re_out = self.ae(attr_mtx)

        d_g_out = self.diff_gnn(ae_out, diff_edge_index)
        o_g_out = self.orgin_gnn(ae_out, edge_index, edge_type)

        emb = torch.stack([ae_out, d_g_out, o_g_out], dim=1)
        emb, _ = self.att_fusion(emb)

        triples = triples.long()
        heads = triples[:, 0]
        tails = triples[:, 1]

        # drug_pair = torch.hstack([emb[heads], emb[tails]])
        # drug_pair = torch.mul(emb[heads],emb[tails])
        # 哈达玛积
        drug_pair = emb[heads] * emb[tails]
        output = self.predictor(drug_pair)
        return output, re_out


# (No Feature Extraction)不使用自编码器的输出，直接使用原始特征输入到图神经网络。
class Model_wo_feat(nn.Module):
    def __init__(self, args):
        super(Model_wo_feat, self).__init__()
        self.ae = AutoEncoder(input_dim=args.feat_dim, latent_dim=args.out_dim, hidden_dim=args.hidden_dim)
        self.diff_gnn = SimpleGraphSAGE(in_channels=args.feat_dim,
                                        hidden_channels=args.hidden_dim,
                                        out_channels=args.out_dim, dropout=args.dropout)
        self.orgin_gnn = R_GCN(num_feat=args.feat_dim,
                               num_hidden=args.hidden_dim,
                               num_outputs=args.out_dim, dropout=args.dropout,
                               rels=args.rels)  # Topology structure encoder
        self.att_fusion = Attention(args.hidden_dim * 2 + args.hidden_dim + args.out_dim)
        self.predictor = DNN((args.hidden_dim * 2 + args.hidden_dim + args.out_dim), args.rels)

    def forward(self, edge_index, diff_edge_index, edge_type, attr_mtx, triples, *args):
        ae_out, re_out = self.ae(attr_mtx)

        d_g_out = self.diff_gnn(attr_mtx, diff_edge_index)
        o_g_out = self.orgin_gnn(attr_mtx, edge_index, edge_type)

        emb = torch.stack([d_g_out, o_g_out], dim=1)
        emb, _ = self.att_fusion(emb)

        triples = triples.long()
        heads = triples[:, 0]
        tails = triples[:, 1]

        # drug_pair = torch.hstack([emb[heads], emb[tails]])
        # drug_pair = torch.mul(emb[heads],emb[tails])
        drug_pair = (emb[heads] + emb[tails]) / 2
        output = self.predictor(drug_pair)
        return output, re_out


#(No Augmented Graph)不使用差异图神经网络的输出，只使用原始图神经网络的输出。
class Model_wo_diffgraph(nn.Module):
    def __init__(self, args):
        super(Model_wo_diffgraph, self).__init__()
        self.ae = AutoEncoder(input_dim=args.feat_dim, latent_dim=args.out_dim, hidden_dim=args.hidden_dim)
        self.diff_gnn = SimpleGraphSAGE(in_channels=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                                        hidden_channels=args.hidden_dim,
                                        out_channels=args.out_dim, dropout=args.dropout)
        self.orgin_gnn = R_GCN(num_feat=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                               num_hidden=args.hidden_dim,
                               num_outputs=args.out_dim, dropout=args.dropout,
                               rels=args.rels)  # Topology structure encoder
        self.att_fusion = Attention(args.hidden_dim * 2 + args.hidden_dim + args.out_dim)
        self.predictor = DNN((args.hidden_dim * 2 + args.hidden_dim + args.out_dim), args.rels)

    def forward(self, edge_index, diff_edge_index, edge_type, attr_mtx, triples, *args):
        ae_out, re_out = self.ae(attr_mtx)

        d_g_out = self.diff_gnn(ae_out, diff_edge_index)
        o_g_out = self.orgin_gnn(ae_out, edge_index, edge_type)

        emb = torch.stack([ae_out, o_g_out], dim=1)
        emb, _ = self.att_fusion(emb)

        triples = triples.long()
        heads = triples[:, 0]
        tails = triples[:, 1]

        # drug_pair = torch.hstack([emb[heads], emb[tails]])
        drug_pair = (emb[heads] + emb[tails]) / 2
        output = self.predictor(drug_pair)
        return output, re_out



# (No Multi-Relational Graph)不使用原始图神经网络的输出，只使用差异图神经网络的输出。
class Model_wo_relgraph(nn.Module):
    def __init__(self, args):
        super(Model_wo_relgraph, self).__init__()
        self.ae = AutoEncoder(input_dim=args.feat_dim, latent_dim=args.out_dim, hidden_dim=args.hidden_dim)
        self.diff_gnn = SimpleGraphSAGE(in_channels=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                                        hidden_channels=args.hidden_dim,
                                        out_channels=args.out_dim, dropout=args.dropout)
        self.orgin_gnn = R_GCN(num_feat=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                               num_hidden=args.hidden_dim,
                               num_outputs=args.out_dim, dropout=args.dropout,
                               rels=args.rels)  # Topology structure encoder
        self.att_fusion = Attention(args.hidden_dim * 2 + args.hidden_dim + args.out_dim)
        self.predictor = DNN((args.hidden_dim * 2 + args.hidden_dim + args.out_dim), args.rels)

    def forward(self, edge_index, diff_edge_index, edge_type, attr_mtx, triples, *args):
        ae_out, re_out = self.ae(attr_mtx)

        d_g_out = self.diff_gnn(ae_out, diff_edge_index)
        o_g_out = self.orgin_gnn(ae_out, edge_index, edge_type)

        emb = torch.stack([ae_out, d_g_out], dim=1)
        emb, _ = self.att_fusion(emb)

        triples = triples.long()
        heads = triples[:, 0]
        tails = triples[:, 1]

        # drug_pair = torch.hstack([emb[heads], emb[tails]])
        drug_pair = (emb[heads] + emb[tails]) / 2
        output = self.predictor(drug_pair)
        return output, re_out



#(No Attention Fusion)不使用注意力融合层，直接将自编码器、差异图神经网络和原始图神经网络的输出进行拼接
class Model_wo_att(nn.Module):
    def __init__(self, args):
        super(Model_wo_att, self).__init__()
        self.ae = AutoEncoder(input_dim=args.feat_dim, latent_dim=args.out_dim, hidden_dim=args.hidden_dim)
        self.diff_gnn = SimpleGraphSAGE(in_channels=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                                        hidden_channels=args.hidden_dim,
                                        out_channels=args.out_dim, dropout=args.dropout)
        self.orgin_gnn = R_GCN(num_feat=args.hidden_dim * 2 + args.hidden_dim + args.out_dim,
                               num_hidden=args.hidden_dim,
                               num_outputs=args.out_dim, dropout=args.dropout,
                               rels=args.rels)  # Topology structure encoder
        self.att_fusion = Attention(args.hidden_dim * 2 + args.hidden_dim + args.out_dim)
        self.predictor = DNN((args.hidden_dim * 2 + args.hidden_dim + args.out_dim) * 6, args.rels)

    def forward(self, edge_index, diff_edge_index, edge_type, attr_mtx, triples, *args):
        ae_out, re_out = self.ae(attr_mtx)

        d_g_out = self.diff_gnn(ae_out, diff_edge_index)
        o_g_out = self.orgin_gnn(ae_out, edge_index, edge_type)

        # emb = torch.stack([ae_out, d_g_out, o_g_out], dim=1)
        emb = torch.hstack([ae_out, d_g_out, o_g_out])
        # emb, _ = self.att_fusion(emb)

        triples = triples.long()
        heads = triples[:, 0]
        tails = triples[:, 1]

        # drug_pair = torch.hstack([emb[heads], emb[tails]])
        drug_pair = (emb[heads] + emb[tails]) / 2
        output = self.predictor(drug_pair)
        return output, re_out


class Model1(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.diff_gnn = SimpleGraphSAGE(in_channels=args.feat_dim,hidden_channels=args.hidden_dim, out_channels=args.out_dim, dropout=args.dropout)
        self.orgin_gnn = R_GCN(num_feat=args.feat_dim,
                               num_hidden=args.hidden_dim,
                               num_outputs=args.out_dim, dropout=args.dropout,
                               rels=args.rels)  # Topology structure encoder
        self.predictor = DNN((args.hidden_dim * 2 + args.hidden_dim + args.out_dim) * 2, args.rels)

    def forward(self, edge_index, diff_edge_index, edge_type, attr_mtx, triples, *args):
        # ae_out, re_out = self.ae(attr_mtx)

        d_g_out = self.diff_gnn(attr_mtx, edge_index)
        # o_g_out = self.orgin_gnn(attr_mtx, edge_index, edge_type)

        emb = d_g_out

        triples = triples.long()
        heads = triples[:, 0]
        tails = triples[:, 1]

        drug_pair = torch.hstack([emb[heads], emb[tails]])
        # drug_pair = torch.mul(emb[heads],emb[tails])
        output = self.predictor(drug_pair)
        return output

#生成图的边索引和边类型张量
def get_edge_index(edge_list, edge_types):
    edge_list_image = edge_list[:, [1, 0]] # 生成反向边，将边 (u, v) 反向为 (v, u)，模拟无向图的双向关系
    edge_index = np.vstack([edge_list, edge_list_image]) # 合并原始边与反向边
    edge_index = torch.tensor(edge_index, dtype=torch.long)  # 将NumPy数组转为PyTorch张量，便于GPU计算。
    edge_type = np.concatenate([edge_types, edge_types])  #为反向边赋予与原始边相同的类型。
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    return edge_index.t().contiguous(), edge_type
