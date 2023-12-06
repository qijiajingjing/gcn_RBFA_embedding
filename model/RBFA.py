import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GCN import GraphConvolution
from model.FAP import FAttentionPooling, global_max_pool, global_avg_pool

class RBFAgcn(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=1):
        super(RBFAgcn, self).__init__()
        self.input_dim = input_dim

        # 第一个gcn层 (N,input_dim) ->(N,hidden_dim) N为所有节点数
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        # 第一个池化层
        self.pool1 = FAttentionPooling(hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool2 = FAttentionPooling(hidden_dim)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool3 = FAttentionPooling(hidden_dim)
        self.gcn4 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool4 = FAttentionPooling(hidden_dim)
        self.gcn5 = GraphConvolution(hidden_dim,hidden_dim)
        self.pool5 = FAttentionPooling(hidden_dim)
        self.gcn6 = GraphConvolution(hidden_dim,hidden_dim)
        self.pool6 = FAttentionPooling(hidden_dim)
        self.gcn7 = GraphConvolution(hidden_dim,hidden_dim)
        self.pool7 = FAttentionPooling(hidden_dim)
        # 把最后的几个全连接层和激活函数 封装在一起
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
#            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
#            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2,num_classes))

    def forward(self, adjacency, input_feature, graph_indicator,atten_score):
        gcn1 = F.relu(self.gcn1(adjacency, input_feature))
        pool1, pool1_graph_indicator, pool1_adjacency = \
            self.pool1(adjacency, gcn1, graph_indicator, atten_score)
        global_pool1 = torch.cat(
            [global_avg_pool(pool1, pool1_graph_indicator),
             global_max_pool(pool1, pool1_graph_indicator)], dim=1)

        gcn2 = F.relu(self.gcn2(adjacency, gcn1)+gcn1)
        pool2, pool2_graph_indicator, pool2_adjacency = \
            self.pool2(adjacency, gcn2, graph_indicator, atten_score)
        global_pool2 = torch.cat(
            [global_avg_pool(pool2, pool2_graph_indicator),
             global_max_pool(pool2, pool2_graph_indicator)],
            dim=1)

        gcn3 = F.relu(self.gcn3(adjacency, gcn2)+gcn2)
        pool3, pool3_graph_indicator, pool3_adjacency = \
            self.pool3(adjacency, gcn3, graph_indicator, atten_score)
        global_pool3 = torch.cat(
            [global_avg_pool(pool3, pool3_graph_indicator),
             global_max_pool(pool3, pool3_graph_indicator)],dim=1)

        gcn4 = F.relu(self.gcn4(adjacency, gcn3)+gcn3)
        pool4, pool4_graph_indicator, pool4_adjacency = \
            self.pool4(adjacency, gcn4, graph_indicator, atten_score)
        global_pool4 = torch.cat(
            [global_avg_pool(pool4, pool4_graph_indicator),
             global_max_pool(pool4, pool4_graph_indicator)],dim=1)

        gcn5 = F.relu(self.gcn5(adjacency, gcn4)+gcn4)
        pool5, pool5_graph_indicator, pool5_adjacency = self.pool5(adjacency, gcn5, graph_indicator, atten_score)
        global_pool5 = torch.cat( [global_avg_pool(pool5, pool5_graph_indicator), global_max_pool(pool5,pool5_graph_indicator)],dim=1)

        gcn6 = F.relu(self.gcn6(adjacency, gcn5)+gcn5)
        pool6, pool6_graph_indicator, pool6_adjacency = self.pool6(adjacency, gcn6, graph_indicator, atten_score)
        global_pool6 = torch.cat( [global_avg_pool(pool6, pool6_graph_indicator), global_max_pool(pool6,pool6_graph_indicator)],dim=1)

        gcn7 = F.relu(self.gcn7(adjacency, gcn6)+gcn6)
        pool7, pool7_graph_indicator, pool7_adjacency = self.pool7(adjacency, gcn7, graph_indicator, atten_score)
        global_pool7 = torch.cat( [global_avg_pool(pool7, pool7_graph_indicator), global_max_pool(pool7,pool7_graph_indicator)],dim=1)

        readout = global_pool1 + global_pool2 + global_pool3 + global_pool4 + global_pool5 + global_pool6 + global_pool7

        logits = self.mlp(readout)
        return logits
