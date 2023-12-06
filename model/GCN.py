import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import scipy.sparse as sp

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """Graph convolution：L*X*\theta
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        # weight matrix
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        ''' use custom parameter initialization method'''
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """Adjacency matrix is a sparse matrix, so sparse matrix multiplication is used in calculation
        Args:
            adjacency (N,N): normalized Laplace matrix
            input_feature（N,input_dim）: N is the number of all nodes (including all graphs)
        """
        support = torch.mm(input_feature, self.weight)  # XW (N,output_dim=hidden_dim)
        output = torch.sparse.mm(adjacency, support)  # L(XW)  (N,output_dim=hidden_dim)
        if self.use_bias:
            output += self.bias
        return output  # (N,output_dim=hidden_dim)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


def tensor_from_numpy(x, device):
    '''Numpy array is converted to tensor and transferred to the device used'''
    return torch.from_numpy(x).to(device)


def normalization(adjacency):
    """Calculation  L=D^-0.5 * (A+I) * D^-0.5,
    Args:
        adjacency: sp.csr_matrix.
    Returns:
        Normalized adjacency matrix, type is torch.sparse.FloatTensor
    """
    adjacency += sp.eye(adjacency.shape[0])  # add self connection: A+I
    degree = np.array(adjacency.sum(1))  # sum the matrix by row
    d_hat = sp.diags(np.power(degree, -0.5).flatten())  # convert to degree matrix
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()  # laplace matrix is converted to coo sparse scheme

    # convert to torch.sparse.FloatTensor
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long() # coordinates of non-zero values of sparse matrix
    values = torch.from_numpy(L.data.astype(np.float32)) #non-zero values
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)
    return tensor_adjacency

def filter_adjacency(adjacency, mask):
    """Update the graph structure according to the mask mask
    Args:
        adjacency: torch.sparse.FloatTensor, adjacency matrix before pooling
        mask: torch.Tensor(dtype=torch.bool), mask vector of nodes
    Returns:
        torch.sparse.FloatTensor, normalized adjacency matrix after pooling
    """
    device = adjacency.device
    mask = mask.cpu().numpy()
    indices = adjacency.coalesce().indices().cpu().numpy()
    num_nodes = adjacency.size(0)
    row, col = indices
    maskout_self_loop = row != col
    row = row[maskout_self_loop]
    col = col[maskout_self_loop]
    sparse_adjacency = sp.csr_matrix((np.ones(len(row)), (row, col)),
                                     shape=(num_nodes, num_nodes), dtype=np.float32)
    filtered_adjacency = sparse_adjacency[mask, :][:, mask]
    return normalization(filtered_adjacency).to(device)