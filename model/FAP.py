import torch
import torch.nn as nn
import torch_scatter  # 注意：torch_scatter 安装时编译需要用到cuda
from model.GCN import GraphConvolution, filter_adjacency

class FAttentionPooling(nn.Module):
    def __init__(self, input_dim):  #keep_ratio, activation=torch.tanh):
        super(FAttentionPooling, self).__init__()
        self.input_dim = input_dim  # (hidden_dim*3)

        #self.keep_ratio = keep_ratio
        #self.activation = activation
        # attention gcn层  (N,hidden_dim*3) -> (N,1)  N个节点 (N,1) bool Flase True C False True
        #self.attn_gcn = GraphConvolution(input_dim, 1)

    def forward(self, adjacency, input_feature, graph_indicator, atten_score):
        # 通过attention gcn层计算注意力分数
        # adjacency拉普拉斯矩阵(N*N)
        # input_feature三个gcn层计算结果通过relu后再拼接 (N,hidden_dim*3)
        # attn_score (N,)
        #attn_score = self.attn_gcn(adjacency, input_feature).squeeze()
        # 通过tanh激活函数 (N,)
        #attn_score = self.activation(attn_score)
        # 强调：attn_score = (N,1)维度 每一个数若C表示0 若N表示1 torch.Tensor
        # 节点掩码向量 (N,)
        #mask = top_select(atten_score, graph_indicator)
        # [bool] (N,1)
        mask = atten_score
        hidden = input_feature[mask]
        # 保留的节点属于哪个图
        mask_graph_indicator = graph_indicator[mask]
        # 得到新的图结构  的邻接矩阵
        mask_adjacency = filter_adjacency(adjacency, mask)
        return hidden, mask_graph_indicator, mask_adjacency

def global_max_pool(x, graph_indicator):
    # 对于每个图保留节点的状态向量 按位置取最大值 最后一个图对应一个状态向量
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_max(x, graph_indicator, dim=0, dim_size=num)[0]

def global_avg_pool(x, graph_indicator):
    # 每个图保留节点的状态向量 按位置取平均值 最后一个图对应一个状态向量
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_mean(x, graph_indicator, dim=0, dim_size=num)