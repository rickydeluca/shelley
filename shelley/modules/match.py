import torch
import torch.nn as nn
import torch.nn.functional as F

from shelley.loss import ContrastiveLossWithAttention, PaleMappingLoss
from shelley.utils.lap_solvers import log_sinkhorn, gumbel_sinkhorn
from shelley.utils.sm_solvers.stable_marriage import stable_marriage
from shelley.utils.sm_solvers.greedy_match import greedy_match
from shelley.utils.rewards import reward_general
from torch_geometric.utils import to_dense_adj


def get_matching_module(f_update, cfg):
    if cfg.NAME == 'sgm':
        model = StableGM(f_update=f_update,
                         beta=cfg.BETA,
                         n_sink_iters=cfg.N_SINK_ITERS,
                         tau=cfg.TAU,
                         mask=cfg.MASK)
        
    elif cfg.NAME == 'sigma':
        model = SIGMA(f_update=f_update,
                      tau=cfg.TAU,
                      n_sink_iter=cfg.N_SINK_ITERS,
                      n_samples=cfg.N_SAMPLES,
                      T=cfg.T,
                      miss_match_value=cfg.MISS_MATCH_VALUE)
    elif cfg.NAME == 'palemap':
        model = PaleMappingMlp(
            f_update=f_update,
            embedding_dim=cfg.EMBEDDING_DIM,
            num_hidden_layers=cfg.NUM_LAYERS,
            activate_function=cfg.ACTIVATE_FUNCTION
        )        
    else:
        raise ValueError(f"Invalid matching model: {cfg.MATCHING.MODEL}")
    
    return model

        
class SIGMA(nn.Module):
    def __init__(self, f_update, tau, n_sink_iter, n_samples, T, miss_match_value):
        super(SIGMA, self).__init__()
        self.f_update = f_update
        self.tau = tau
        self.n_sink_iter = n_sink_iter
        self.n_samples = n_samples
        self.T = T
        self.miss_match_value = miss_match_value

    def forward(self, graph_s, graph_t, train=False, train_dict=None):
        loss = 0.0
        loss_count = 0

        best_reward = float('-inf')
        best_logits = None

        n_node_s = graph_s.num_nodes
        n_node_t = graph_t.num_nodes

        theta_previous = None
        for _ in range(self.T):
            # Adjust feature importance
            if theta_previous is not None:
                x_s_importance = theta_previous.sum(-1, keepdims=True)
                x_t_importance = theta_previous.sum(-2, keepdims=True).transpose(-1, -2)
            else:
                x_s_importance = torch.ones([n_node_s, 1]).to(graph_s.x.device)
                x_t_importance = torch.ones([n_node_t, 1]).to(graph_t.x.device)

            # Propagate
            h_s = self.f_update(graph_s.x, graph_s.edge_index, graph_s.edge_attr, x_importance=x_s_importance)
            h_t = self.f_update(graph_t.x, graph_t.edge_index, graph_t.edge_attr, x_importance=x_t_importance)

            # Scale 50.0 allows logits to be larger than the noise
            h_s = h_s / h_s.norm(dim=-1, keepdim=True)
            h_t = h_t / h_t.norm(dim=-1, keepdim=True)
            log_alpha = h_s @ h_t.transpose(-1, -2) * 50.0

            log_alpha_ = F.pad(log_alpha, (0, n_node_s, 0, n_node_t), value=0.0)

            theta_new = gumbel_sinkhorn(log_alpha_, self.tau, self.n_sink_iter, self.n_samples, noise=True)

            r = reward_general(n_node_s, n_node_t,
                               graph_s.edge_index, graph_t.edge_index,
                               theta_new,
                               miss_match_value=self.miss_match_value)

            if best_reward < r.item():
                best_reward = r.item()
                loss = loss - r
                loss_count += 1
                theta_previous = theta_new[:, :n_node_s, :n_node_t].mean(0).detach().clone()
                best_logits = log_alpha_.detach().clone()

        if train:
            return best_logits[:n_node_s, :n_node_t], loss / float(loss_count)
        
        else:
            return best_logits[:n_node_s, :n_node_t]
        

class StableGM(nn.Module):
    def __init__(self, f_update, beta=0.1, n_sink_iters=10, tau=1.0, mask=False):
        super(StableGM, self).__init__()
        self.f_update = f_update
        self.loss_fn = ContrastiveLossWithAttention()
        self.n_sink_iters = n_sink_iters
        self.beta = beta
        self.tau = tau
        self.mask = mask

    def forward(self, graph_s, graph_t, train=False, train_dict=None):
        # Generate embeddings
        h_s = self.f_update(graph_s.x, graph_s.edge_index, graph_s.edge_attr)
        h_t = self.f_update(graph_t.x, graph_t.edge_index, graph_t.edge_attr)
 
        if train:
            # Take only the embeddings of training nodes
            valid_srcs, valid_tgts = train_dict['gt_perm'].squeeze(0).nonzero(as_tuple=True)
            h_s = h_s[valid_srcs]
            h_t = h_t[valid_tgts]

            # Generate corresponding groundtruth
            gt = torch.eye(valid_srcs.size(0)).unsqueeze(0).to(valid_srcs.device)
            src_ns = torch.tensor([valid_srcs.size(0)]).to(valid_srcs.device)
            tgt_ns = torch.tensor([valid_tgts.size(0)]).to(valid_tgts.device)

            if h_s.dim() == h_t.dim() == 2:
                h_s = h_s.unsqueeze(0)
                h_t = h_t.unsqueeze(0)

            # Cosine similarity
            h_s = h_s / h_s.norm(dim=-1, keepdim=True)
            h_t = h_t / h_t.norm(dim=-1, keepdim=True)
            sim_mat = torch.matmul(h_s, h_t.transpose(1, 2))

            # Sinkhorn ranking
            rank_mat = log_sinkhorn(sim_mat, n_iter=self.n_sink_iters, tau=self.tau)

            # Hardness attention loss
            loss = self.loss_fn(rank_mat,
                                gt, 
                                src_ns, 
                                tgt_ns,
                                self.beta,
                                mask=self.mask)
            
            return sim_mat, loss
        
        else:
            if h_s.dim() == h_t.dim() == 2:
                h_s = h_s.unsqueeze(0)
                h_t = h_t.unsqueeze(0)

            # Cosine similarity
            h_s = h_s / h_s.norm(dim=-1, keepdim=True)
            h_t = h_t / h_t.norm(dim=-1, keepdim=True)
            sim_mat = torch.matmul(h_s, h_t.transpose(1, 2))

            return sim_mat
        

class PaleMappingMlp(nn.Module):
    """
    Class to handle both linear and mlp mapping models by specifying the
    number of hidden layers.
    """
    def __init__(self, f_update=None, embedding_dim=256, num_hidden_layers=1, activate_function='sigmoid'):
        super(PaleMappingMlp, self).__init__()
        self.f_update = f_update
        self.source_embedding = None
        self.target_embedding = None
        self.loss_fn = PaleMappingLoss()

        self.num_hidden_layers = num_hidden_layers

        if num_hidden_layers > 0:   # MLP
            if activate_function == 'sigmoid':
                self.activate_function = nn.Sigmoid()
            elif activate_function == 'relu':
                self.activate_function = nn.ReLU()
            else:
                self.activate_function = nn.Tanh()

            hidden_dim = 2 * embedding_dim
            layers = [nn.Linear(embedding_dim, hidden_dim, bias=True), self.activate_function]

            for _ in range(num_hidden_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim, bias=True), self.activate_function])

            layers.append(nn.Linear(hidden_dim, embedding_dim, bias=True))
            self.mapping_network = nn.Sequential(*layers)
        else:   # Linear
            self.mapping_network = nn.Linear(embedding_dim, embedding_dim, bias=True)

    def forward(self, graph_s, graph_t, train=False, train_dict=None):
        # GCN is optional in PaleMapping
        if self.f_update is not None:
            h_s = self.f_update(graph_s.x, graph_s.edge_index, graph_s.edge_attr)
            h_t = self.f_update(graph_t.x, graph_t.edge_index, graph_t.edge_attr)
        else:
            h_s = graph_s.x
            h_t = graph_t.x

        if train:
            # Map features
            source_indices, target_indices = train_dict['gt_perm'].squeeze(0).nonzero(as_tuple=True)
            source_feats = h_s[source_indices]
            target_feats = h_t[target_indices]
            source_feats_after_mapping = self.map(source_feats)

            # Compute euclidean loss
            # batch_size = source_feats.shape[0]
            mapping_loss = self.loss_fn.loss(source_feats_after_mapping, target_feats)

            return None, mapping_loss
        else:
            # Map features
            source_feats_after_mapping = self.map(h_s)
            target_feats = h_t

            if source_feats_after_mapping.dim() == target_feats.dim() == 2:
                source_feats_after_mapping = source_feats_after_mapping.unsqueeze(0)
                target_feats = target_feats.unsqueeze(0)

            # Cosine similarity
            source_feats_after_mapping = source_feats_after_mapping / source_feats_after_mapping.norm(dim=-1, keepdim=True)
            target_feats = target_feats / target_feats.norm(dim=-1, keepdim=True)
            sim_mat = torch.matmul(source_feats_after_mapping, target_feats.transpose(1, 2))
            return sim_mat
        

    def map(self, source_feats):
        if self.num_hidden_layers > 0:
            ret = self.mapping_network(source_feats)
        else:
            ret = self.mapping_network(source_feats)
        ret = F.normalize(ret, dim=1)
        return ret