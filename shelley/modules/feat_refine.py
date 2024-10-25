from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Sequential, Tanh
from torch_geometric.nn import MLP, GCNConv, GINEConv
from torch_geometric.typing import SparseTensor


def get_feat_refine_module(cfg):
    if cfg.NAME == 'gcn':
        f_update = GCN(
            in_channels=cfg.IN_CHANNELS,
            hidden_channels=cfg.HIDDEN_CHANNELS,
            out_channels=cfg.OUT_CHANNELS,
            num_layers=cfg.NUM_LAYERS
        )
    elif cfg.NAME == 'gine':
        f_update = GINE(
            in_channels=cfg.IN_CHANNELS,
            dim=cfg.DIM,
            out_channels=cfg.OUT_CHANNELS,
            num_conv_layers=cfg.NUM_CONV_LAYERS
        )
    elif cfg.NAME == 'mlp':
        f_update = MLPEmb(
            in_channels=cfg.IN_CHANNELS,
            hidden_channels=cfg.HIDDEN_CHANNELS,
            out_channels=cfg.OUT_CHANNELS,
            num_layers=cfg.NUM_LAYERS
        )
    elif cfg.NAME == 'none':
        return None
    else:
        raise ValueError(f"Invalid embedding model: {cfg.EMBEDDING.MODEL}")
    
    return f_update
    

class GINE(nn.Module):
    def __init__(self, in_channels, out_channels, dim, num_conv_layers=1, bias=True):
        super(GINE, self).__init__()

        act = Tanh()
        eps = 0.0
        train_eps = False

        self.bn_in = nn.BatchNorm1d(in_channels)

        self.num_conv_layers = num_conv_layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # First conv layer
        nn1 = Sequential(Linear(in_channels, dim, bias=bias), act)
        self.conv_layers.append(GINEConv(nn1, eps=eps, train_eps=train_eps, edge_dim=in_channels))
        self.bn_layers.append(nn.BatchNorm1d(dim))

        # Remaning conv layers
        for _ in range(1, num_conv_layers):
            nn_hid = Sequential(Linear(dim, dim, bias=bias), act)
            self.conv_layers.append(GINEConv(nn_hid, eps=eps, train_eps=train_eps, edge_dim=in_channels))
            self.bn_layers.append(nn.BatchNorm1d(dim))

        # Final linar layer
        self.fc = Linear(dim, out_channels, bias=False)

    def forward(self, x, edge_index, edge_attr, x_importance=None):
        if x_importance is not None:
            x = x * x_importance

        x = self.bn_in(x)

        for i in range(self.num_conv_layers):
            x = self.bn_layers[i](self.conv_layers[i](x, edge_index, edge_attr))

        # Linear
        x = torch.tanh(self.fc(x))
        
        return x
    

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, normalize=True, bias=True):
        super(GCN, self).__init__()

        self.conv_layers = nn.ModuleList()

        if num_layers == 1:
            self.conv_layers.append(GCNConv(in_channels, out_channels, normalize=normalize, bias=bias))
        else:
            self.conv_layers.append(GCNConv(in_channels, hidden_channels, normalize=normalize, bias=bias))
            for _ in range(1, num_layers - 1):
                self.conv_layers.append(GCNConv(hidden_channels, hidden_channels, normalize=normalize, bias=bias))
            self.conv_layers.append(GCNConv(hidden_channels, out_channels, normalize=normalize, bias=bias))

    def forward(self, x: Tensor, edge_index: Union[Tensor, SparseTensor],
                edge_attr: Optional[Tensor] = None, x_importance=None) -> Tensor:
        
        if x_importance is not None:
            x = x * x_importance
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index, edge_attr)
            x = F.relu(x)

        return x

class MLPEmb(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, bias=True):
        super(MLPEmb, self).__init__()
        self.mlp = MLP(in_channels=in_channels,
                       hidden_channels=hidden_channels,
                       out_channels=out_channels,
                       num_layers=num_layers,
                       bias=bias)

    def forward(self, x: Tensor, edge_index: Union[Tensor, SparseTensor],
                edge_attr: Optional[Tensor] = None, x_importance=None) -> Tensor:
        
        if x_importance is not None:
            x = x * x_importance
        
        x = self.mlp(x)

        return x

