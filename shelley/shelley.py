import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from shelley.modules.feat_init import get_feat_init_module
from shelley.modules.feat_refine import get_feat_refine_module
from shelley.modules.match import get_matching_module

from shelley.data.utils import dict_to_perm_mat, move_tensors_to_device
from shelley.evaluation.matchers import greedy_match
from shelley.evaluation.metrics import compute_accuracy
        

class SHELLEY_SoG(nn.Module):
    def __init__(self, cfg):
        """
        Split-on-graphs mode.
        """
        super(SHELLEY_SoG, self).__init__()
        
        # Init modules
        self.f_init = get_feat_init_module(cfg.FEAT_INIT)
        self.f_update = get_feat_refine_module(cfg.FEAT_REFINE)
        self.model = get_matching_module(self.f_update, cfg.MATCHING)

        # Configure training
        self.epochs = cfg.TRAIN.EPOCHS
        self.patience = cfg.TRAIN.PATIENCE

        if cfg.TRAIN.OPTIMIZER == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=cfg.TRAIN.LR,
                weight_decay=cfg.TRAIN.L2NORM
            )
        elif cfg.TRAIN.OPTIMIZER == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=cfg.TRAIN.LR,
                momentum=cfg.TRAIN.MOMENTUM)
        else:
            raise ValueError(f"Invalid optimizer: {cfg.TRAIN.OPTIMIZER}")

    def align(self, pair_dict, ret_loss=False):
        """
        Predict alignment between a graph pair.
        """
        pair_dict = move_tensors_to_device(pair_dict, self.device)
        
        # Read input
        self.graph_s = pair_dict['graph_pair'][0]
        self.graph_t = pair_dict['graph_pair'][1]
        self.src_ns = torch.tensor([self.graph_s.num_nodes]).to(self.device)
        self.tgt_ns = torch.tensor([self.graph_t.num_nodes]).to(self.device)
        self.gt_full = dict_to_perm_mat(pair_dict['gt_full'], self.graph_s.num_nodes, self.graph_t.num_nodes).to(self.device).unsqueeze(0)

        # Init features
        self.init_features()

        # Predict alignment
        if ret_loss:
            train_dict = {'gt_perm': self.gt_full, 'src_ns': self.src_ns, 'tgt_ns': self.tgt_ns}
            S, loss = self.model(self.graph_s, self.graph_t, train=True, train_dict=train_dict)
            return S, loss
        
        # Evaluation
        else:
            S = self.model(self.graph_s, self.graph_t, train=False)
            return S

    def init_features(self):
        self.graph_s.x = self.f_init(self.graph_s).to(self.device)
        self.graph_t.x = self.f_init(self.graph_t).to(self.device)

        # Extend dimension of edge attributes also
        if self.graph_s.x.size(1) != self.graph_s.edge_attr.size(1):
            self.graph_s.edge_attr = self.graph_s.edge_attr.repeat(1, self.graph_s.x.size(1))
        
        if self.graph_t.x.size(1) != self.graph_t.edge_attr.size(1):
            self.graph_t.edge_attr = self.graph_t.edge_attr.repeat(1, self.graph_t.x.size(1))
        
    def train_eval(self, train_loader, val_loader=None, device=None, verbose=False):
        """
        Train and validate model.
        """
        self.verbose = verbose
        self.device = device
        validate = True if val_loader is not None else False
        best_val_loss = float('inf')
        best_state_dict = {}
        best_epoch = -1
        patience_counter = 0

        for epoch in range(self.epochs):
            print(f"Epoch: {epoch+1}/{self.epochs}")

            # Train
            train_loss = self.train(train_loader)

            # Eval
            if validate:
                val_loss = self.evaluate(val_loader)
                
                print(f"Epoch: {epoch+1}/{self.epochs}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
                
                # Update best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = self.model.state_dict()
                    best_epoch = epoch+1
                    patience_counter = 0
                else:
                    patience_counter += 1
            
                # Check for early stop
                if patience_counter > self.patience:
                    print(f"Early stop triggered after {epoch+1} epochs!")
                    break
            
            else:
                print(f"Epoch: {epoch+1}/{self.epochs}, Train loss: {train_loss:.4f}")

        # Load best state dict
        if validate:
            self.model.load_state_dict(best_state_dict)

    def train(self, train_loader):
        """
        Train model.
        """
        self.model.train()
        n_batches = len(train_loader)
        train_loss = 0
        for pair_dict in tqdm(train_loader, desc="Training"):
            
            # Foraward step
            _, loss = self.align(pair_dict, ret_loss=True)

            # Backward step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss
        
        return train_loss / n_batches
    
    @torch.no_grad()
    def evaluate(self, eval_loader, use_acc=False):
        """
        Evaluate model (validation/test).
        """
        self.model.eval()
        n_batches = len(eval_loader)
        if use_acc:
            eval_accs = []
            for pair_dict in tqdm(eval_loader, desc='Testing'):
                S = self.align(pair_dict, ret_loss=False).squeeze(0).detach().cpu().numpy()
                P = greedy_match(S)
                gt_test = dict_to_perm_mat(pair_dict['gt_full'], pair_dict['graph_pair'][0].num_nodes, pair_dict['graph_pair'][1].num_nodes).detach().cpu().numpy()
                eval_acc = compute_accuracy(P, gt_test)
                eval_accs.append(eval_acc)

            return np.mean(eval_accs), np.std(eval_accs)

        else:
            eval_loss = 0
            for pair_dict in tqdm(eval_loader, desc='Validation'):
                _, loss = self.align(pair_dict, ret_loss=True)
                eval_loss += loss.item()
            
            return eval_loss / n_batches

    def pprint(self, str):
        if self.verbose:
            print(str)


class SHELLEY_SoN(nn.Module):
    def __init__(self, cfg):
        """
        Split-on-nodes mode.
        """
        super(SHELLEY_SoN, self).__init__()
        
        # Init modules
        self.f_init = get_feat_init_module(cfg.FEAT_INIT)
        self.f_update = get_feat_refine_module(cfg.FEAT_REFINE)
        self.model = get_matching_module(self.f_update, cfg.MATCHING)

        # Configure training
        self.epochs = cfg.TRAIN.EPOCHS
        self.patience = cfg.TRAIN.PATIENCE

        if cfg.TRAIN.OPTIMIZER == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=cfg.TRAIN.LR,
                weight_decay=cfg.TRAIN.L2NORM
            )
        elif cfg.TRAIN.OPTIMIZER == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=cfg.TRAIN.LR,
                momentum=cfg.TRAIN.MOMENTUM)
        else:
            raise ValueError(f"Invalid optimizer: {cfg.TRAIN.OPTIMIZER}")
        
    def forward(self, pair_dict, verbose=False):
        # Read input
        self.graph_s = pair_dict['graph_pair'][0]
        self.graph_t = pair_dict['graph_pair'][1]
        self.device = self.graph_s.edge_index.device
        self.src_ns = torch.tensor([self.graph_s.num_nodes]).to(self.device)
        self.tgt_ns = torch.tensor([self.graph_t.num_nodes]).to(self.device)
        self.gt_train = dict_to_perm_mat(pair_dict['gt_train'], self.graph_s.num_nodes, self.graph_t.num_nodes).to(self.device).unsqueeze(0)
        self.gt_val = dict_to_perm_mat(pair_dict['gt_val'], self.graph_s.num_nodes, self.graph_t.num_nodes).to(self.device).unsqueeze(0)
        self.verbose = verbose

        # Perform validation only if `gt_val` contains non-zero elements
        self.validate = torch.any(self.gt_val != 0)

        # Init features
        self.init_features()
        
        # Train model
        self.best_val_epoch = self.train_eval()

        # Get alignment
        self.S = self.get_alignment()

        return self.S, self.best_val_epoch

    def init_features(self):
        device = self.graph_s.edge_index.device
        self.graph_s.x = self.f_init(self.graph_s).to(device)
        self.graph_t.x = self.f_init(self.graph_t).to(device)

        # Extend dimension of edge attributes also
        if self.graph_s.x.size(1) != self.graph_s.edge_attr.size(1):
            self.graph_s.edge_attr = self.graph_s.edge_attr.repeat(1, self.graph_s.x.size(1))
        
        if self.graph_t.x.size(1) != self.graph_t.edge_attr.size(1):
            self.graph_t.edge_attr = self.graph_t.edge_attr.repeat(1, self.graph_t.x.size(1))
        
    def train_eval(self):
        best_val_loss = float('inf')
        best_val_epoch = 0
        best_state_dict = {}
        patience_counter = 0

        for epoch in range(self.epochs):
            # Train
            train_loss = self.train()

            # Eval
            if self.validate:
                val_loss = self.evaluate()

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch + 1
                    best_state_dict = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Check for early stop
                if patience_counter > self.patience:
                    print(f"Early stop triggered after {epoch+1} epochs!")
                    break

                self.pprint(f"Epoch: {epoch+1}, Train loss: {train_loss}, Val loss: {val_loss}")
            else:
                self.pprint(f"Epoch: {epoch+1}, Train loss: {train_loss}")
            

        # Load best model
        if best_state_dict:
            self.model.load_state_dict(best_state_dict)

        return best_val_epoch


    def train(self):
        self.model.train()

        # Forward
        train_dict = {'gt_perm': self.gt_train, 'src_ns': self.src_ns, 'tgt_ns': self.tgt_ns}
        _, train_loss = self.model(self.graph_s, self.graph_t, train=True, train_dict=train_dict)

        # Backward
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()

        return train_loss
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        train_dict = {'gt_perm': self.gt_val, 'src_ns': self.src_ns, 'tgt_ns': self.tgt_ns}
        _, val_loss = self.model(self.graph_s, self.graph_t, train=True, train_dict=train_dict)
        return val_loss
    
    @torch.no_grad()
    def get_alignment(self):
        self.model.eval()
        S = self.model(self.graph_s, self.graph_t).squeeze(0).detach().cpu().numpy()
        return S
    
    @torch.no_grad()
    def get_embeddings(self):
        h_s = self.model.f_update(self.graph_s.x,
                                  self.graph_s.edge_index,
                                  edge_attr=self.graph_s.edge_attr).detach()
        h_t = self.model.f_update(self.graph_t.x,
                                  self.graph_t.edge_index,
                                  edge_attr=self.graph_t.edge_attr).detach()
        return h_s, h_t

    def pprint(self, str):
        if self.verbose:
            print(str)