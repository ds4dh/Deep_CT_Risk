import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool


class GraphCondGlobal(nn.Module):
    """
    Graph-based (as opposed to node-based) conditional (on trial phase and clinical condition)
    GNN (using graph-conv layers) with global pooling (as opposed to selective pooling).
    """
    def __init__(self, config_network):
        super(GraphCondGlobal, self).__init__()
        # torch.manual_seed(12345)

        conv_type = config_network["conv"]
        d_cond = config_network["d_cond"]
        n_c = config_network["n_c"]
        d_in = config_network["d_in"]
        d_h = config_network["d_h"]
        self.dropout = config_network["dropout"]
        if 'gcn' in conv_type.lower():
            self.conv1 = GCNConv(d_in, d_h)
            self.conv2 = GCNConv(d_h, d_h)
            self.conv3 = GCNConv(d_h, d_h)
            self.conv4 = GCNConv(d_h, d_h)

        elif 'gat' in conv_type.lower():
            self.conv1 = GATConv(d_in, d_h)
            self.conv2 = GATConv(d_h, d_h)
            self.conv3 = GATConv(d_h, d_h)
            self.conv4 = GATConv(d_h, d_h)

        elif 'sage' in conv_type.lower():
            self.conv1 = SAGEConv(d_in, d_h)
            self.conv2 = SAGEConv(d_h, d_h)
            self.conv3 = SAGEConv(d_h, d_h)
            self.conv4 = SAGEConv(d_h, d_h)

        self.lin = nn.Linear(d_h + d_cond, n_c)

    def forward(self, x, edge_index, batch, cond):
        x = self.conv1(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()
        x = self.conv4(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat((x, cond), dim=-1)
        x = self.lin(x)

        return x


class GraphCondSelective(nn.Module):
    """
    Graph-based (as opposed to node-based) conditional (on trial phase and clinical condition)
    GNN (using graph-conv layers) with selective pooling (as opposed to global pooling).
    """
    def __init__(self, config_network):
        super(GraphCondSelective, self).__init__()

        conv_type = config_network["conv"]
        self.d_latent = config_network["d_latent"]  # 20
        self.len_base_nodes_list = len(config_network["pool"]["base_nodes_list"])

        d_cond = config_network["d_cond"]
        n_c = config_network["n_c"]
        d_in = config_network["d_in"]
        d_h = config_network["d_h"]
        self.dropout = config_network["dropout"]

        if 'gcn' in conv_type.lower():
            self.conv1 = GCNConv(d_in, d_h)
            self.conv2 = GCNConv(d_h, d_h)
            self.conv3 = GCNConv(d_h, d_h)
            self.conv4 = GCNConv(d_h, d_h)
            self.conv5 = GCNConv(d_h, self.d_latent)

        elif 'gat' in conv_type.lower():
            self.conv1 = GATConv(d_in, d_h)
            self.conv2 = GATConv(d_h, d_h)
            self.conv3 = GATConv(d_h, d_h)
            self.conv4 = GATConv(d_h, d_h)
            self.conv5 = GATConv(d_h, self.d_latent)

        elif 'sage' in conv_type.lower():
            self.conv1 = SAGEConv(d_in, d_h)
            self.conv2 = SAGEConv(d_h, d_h)
            self.conv3 = SAGEConv(d_h, d_h)
            self.conv4 = SAGEConv(d_h, d_h)
            self.conv5 = SAGEConv(d_h, self.d_latent)

        self.lin1 = nn.Linear(self.d_latent * (1+self.len_base_nodes_list), d_h)
        self.lin2 = nn.Linear(d_h+d_cond, n_c)

        self.register_buffer('latent_base', None, persistent=False)
        self.register_buffer('latent_global', None, persistent=False)

        self.gradients = None

    def hooker(self, grad):
        self.gradients = grad

    def forward(self, x, edge_index, batch, base, cond):
        x = self.conv1(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()

        x = self.conv3(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()

        x = self.conv4(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()

        x = self.conv5(x, edge_index)

        x_g = global_mean_pool(x, batch)
        x_s = selective_pool(x, batch, base)
        x = torch.cat((x_s, x_g), dim=1)

        h = x.register_hook(self.hooker)

        self.register_buffer('latent_base', x.view(x.shape[0], 1+self.len_base_nodes_list, self.d_latent),
                             persistent=False)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin1(x)

        self.register_buffer('latent_global', x, persistent=False)

        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()
        x = torch.cat((x, cond), dim=-1)
        x = self.lin2(x)

        return x


def get_indices(nodes, base_list):
    return list(map(lambda n: nodes.index(n) if n in nodes else None, base_list.copy()))


def get_indices_batch(nodes_batch, base_list):
    return list(map(lambda nodes: get_indices(nodes, base_list), nodes_batch))


def selective_pool(inp_x, inp_batch, base):
    b, d = len(base), inp_x.shape[1]
    x = torch.zeros(b, len(base[0]), d, device=inp_x.device, dtype=inp_x.dtype)
    for i_b in range(b):
        _ind = [base[i_b].index(_i) for _i in base[i_b] if _i]
        x[i_b, _ind, :] = inp_x[inp_batch == i_b][[base[i_b][_i] for _i in _ind], :]

    return x.view(b, -1)
