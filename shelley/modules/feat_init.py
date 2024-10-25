import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import degree
from tqdm import tqdm
from shelley.loss import PaleEmbeddingLoss
import torch.nn.functional as F

def get_feat_init_module(cfg):
    # Feature initialization
    if cfg.NAME.lower() == 'degree':
        f_init = Degree()
    elif cfg.NAME.lower() == 'shared':
        f_init = Shared(cfg.FEATURE_DIM)
    elif cfg.NAME.lower() == 'paleemb':
        f_init = PaleEmbedding(cfg)
    else:
        raise ValueError(f"Invalid features: {cfg.TYPE}.")
    
    return f_init


class Degree(nn.Module):
    def __init__(self):
        super(Degree, self).__init__()

    def forward(self, graph):
        return degree(graph.edge_index[0], num_nodes=graph.num_nodes).unsqueeze(1)
    

class Shared(nn.Module):
    def __init__(self, node_feature_dim):
        super(Shared, self).__init__()
        self.node_feature_dim = node_feature_dim

    def forward(self, graph):
        return torch.ones((graph.num_nodes, self.node_feature_dim))
    

class _PaleEmbedding(nn.Module):
    def __init__(self, n_nodes, embedding_dim, deg, neg_sample_size, cuda):
        super(_PaleEmbedding, self).__init__()
        self.node_embedding = nn.Embedding(n_nodes, embedding_dim)
        self.deg = deg
        self.neg_sample_size = neg_sample_size
        self.link_pred_layer = PaleEmbeddingLoss()
        self.n_nodes = n_nodes
        self.use_cuda = cuda

    @staticmethod
    def fixed_unigram_candidate_sampler(num_sampled, unique, range_max, distortion, unigrams):
        weights = unigrams**distortion
        prob = weights/weights.sum()
        sampled = np.random.choice(range_max, num_sampled, p=prob, replace=~unique)
        return sampled

    def loss(self, nodes, neighbor_nodes):
        batch_output, neighbor_output, neg_output = self.forward(nodes, neighbor_nodes)
        batch_size = batch_output.shape[0]
        loss, loss0, loss1 = self.link_pred_layer.loss(batch_output, neighbor_output, neg_output)
        loss = loss/batch_size
        loss0 = loss0/batch_size
        loss1 = loss1/batch_size
        
        return loss, loss0, loss1


    def forward(self, nodes, neighbor_nodes=None):
        node_output = self.node_embedding(nodes)
        node_output = F.normalize(node_output, dim=1)

        if neighbor_nodes is not None:
            neg = self.fixed_unigram_candidate_sampler(
                num_sampled=self.neg_sample_size,
                unique=False,
                range_max=len(self.deg),
                distortion=0.75,
                unigrams=self.deg
                )

            neg = torch.LongTensor(neg)
            
            if self.use_cuda:
                neg = neg.cuda()
            neighbor_output = self.node_embedding(neighbor_nodes)
            neg_output = self.node_embedding(neg)
            # normalize
            neighbor_output = F.normalize(neighbor_output, dim=1)
            neg_output = F.normalize(neg_output, dim=1)

            return node_output, neighbor_output, neg_output

        return node_output

    def get_embedding(self, batch_size=None):
        nodes = np.arange(self.n_nodes)
        nodes = torch.LongTensor(nodes)
        if self.use_cuda:
            nodes = nodes.cuda()
        embedding = None
        BATCH_SIZE = 512 if batch_size is None else batch_size
        for i in range(0, self.n_nodes, BATCH_SIZE):
            j = min(i + BATCH_SIZE, self.n_nodes)
            batch_nodes = nodes[i:j]
            if batch_nodes.shape[0] == 0: break
            batch_node_embeddings = self.forward(batch_nodes)
            if embedding is None:
                embedding = batch_node_embeddings
            else:
                embedding = torch.cat((embedding, batch_node_embeddings))

        return embedding

class PaleEmbedding(nn.Module):
    def __init__(self, cfg):
        super(PaleEmbedding, self).__init__()
        self.emb_batchsize = cfg.BATCH_SIZE
        self.emb_lr = cfg.LR
        self.neg_sample_size = cfg.NEG_SAMPLE_SIZE
        self.embedding_dim = cfg.EMBEDDING_DIM
        self.emb_epochs = cfg.EPOCHS
        self.embedding_name = cfg.EMBEDDING_NAME
        self.emb_optimizer = cfg.OPTIMIZER
        self.use_cuda = cfg.CUDA

    def forward(self, graph):
        # Init PALE
        num_nodes = graph.num_nodes
        edges = graph.edge_index.t().detach().cpu().numpy()
        deg = degree(graph.edge_index[0],
                     num_nodes=num_nodes).detach().cpu().numpy() 
        return self.learn_embedding(num_nodes, deg, edges)

    def learn_embedding(self, num_nodes, deg, edges):
        # Embedding model
        embedding_model = _PaleEmbedding(n_nodes=num_nodes,
                                        embedding_dim=self.embedding_dim,
                                        deg=deg,
                                        neg_sample_size=self.neg_sample_size,
                                        cuda=self.use_cuda)
        if self.use_cuda:
            embedding_model = embedding_model.cuda()

        # Optimizer
        if self.emb_optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, embedding_model.parameters()),
                lr=self.emb_lr
            )
        else:
            raise ValueError(f'Invalid embedding optimizer: {self.emb_optimizer}.')
        
        # Train
        embedding = self.train_embedding(embedding_model, edges, optimizer)

        return embedding


    def train_embedding(self, embedding_model, edges, optimizer):
        n_iters = len(edges) // self.emb_batchsize
        # assert n_iters > 0, "`batch_size` is too large."
        
        if n_iters == 0:
            n_iters = len(edges)

        if(len(edges) % self.emb_batchsize > 0):
            n_iters += 1
        print_every = int(n_iters/4) + 1
        total_steps = 0
        n_epochs = self.emb_epochs
        for epoch in tqdm(range(1, n_epochs + 1), desc="Init PALE feats:"):
            start = time.time()     # Time evaluation
            
            np.random.shuffle(edges)
            for iter in range(n_iters):
                batch_edges = torch.LongTensor(edges[iter*self.emb_batchsize:(iter+1)*self.emb_batchsize])
                if self.cuda:
                    batch_edges = batch_edges.cuda()
                start_time = time.time()
                optimizer.zero_grad()
                loss, loss0, loss1 = embedding_model.loss(batch_edges[:, 0], batch_edges[:,1])
                loss.backward()
                optimizer.step()
                total_steps += 1
            
            self.embedding_epoch_time = time.time() - start     # Time evaluation
            
        embedding = embedding_model.get_embedding(batch_size=self.emb_batchsize)
        embedding = embedding.cpu().detach().numpy()
        embedding = torch.FloatTensor(embedding)
        if self.cuda:
            embedding = embedding.cuda()

        return embedding