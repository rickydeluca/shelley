import random
from typing import Optional

import torch
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from torch_geometric.utils import degree

import shelley.data.utils as utils


class RealDataset(Dataset):
    def __init__(self,
                 source_path: str,
                 target_path: str,
                 gt_path: Optional[str] = None,
                 gt_mode: Optional[str] = 'homology',
                 train_ratio: Optional[float] = 0.2,
                 val_ratio: Optional[float] = 0.0,
                 seed: Optional[int] = None):
        
        super(RealDataset, self).__init__()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.size = 1       # Only one pair in real datasets
        self.gt_mode = gt_mode
        
        # Load PyG graphs
        self.source_pyg, self.source_id2idx = utils.edgelist_to_pyg(source_path)
        self.target_pyg, self.target_id2idx = utils.edgelist_to_pyg(target_path)
        self.source_idx2id = {idx: id for id, idx in self.source_id2idx.items()}
        self.target_idx2id = {idx: id for id, idx in self.target_id2idx.items()}

        # If the groundtruth is not given, generate it by `gt_mode`
        if gt_path is not None:
            self.gt_dict = self._read_dict(gt_path)
        else:
            if gt_mode == 'homology':
                self.gt_dict = self._generate_homology_gt()
            else:
                self.gt_dict = None

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.get_pair(idx)
    
    def _read_dict(self, path):
        """
        Read the groundtruth file and return it
        as node indices dictionary.
        """
        gt_dict = {}

        with open(path, 'r') as file:
            for line in file:
                s_id, t_id = line.split()
                gt_dict[self.source_id2idx[s_id]] = self.target_id2idx[t_id]

        return gt_dict
    
    def _generate_homology_gt(self):
        """
        Generate a groundtruth dictionary using
        the node names homology.
        """
        gt_dict = {}
        for s_id, s_idx in self.source_id2idx.items():
            if s_id in self.target_id2idx.keys():
                gt_dict[s_idx] = self.target_id2idx[s_id]

        return gt_dict
    
    def _generate_homology_degree_gt(self, top_p=0.2):
        """
        Generate groundtruth dictionary by homology, but
        keep in the train dictionary only the nodes with
        an high degree value (`top_p`).
        """
        train_dict = self._homology_seed_nodes(p=top_p, use_degree=True)
        full_dict = self._generate_homology_gt()
        test_dict = {k: v for k, v in full_dict.items() if k not in train_dict}
        return train_dict, test_dict

    def _homology_seed_nodes(self, p=0.2, use_degree=False):
        """
        Generate a mapping between the nodes of source
        and target networks using the name homology of the nodes.
        Select only the `p` percentage of this mapping as 
        seed nodes for trainable algorithms.
        """

        num_seeds = int(self.source_pyg.num_nodes * p)

        if self.seed is not None:
            random.seed(self.seed)

        if use_degree:
            source_degrees = degree(self.source_pyg.edge_index[0])
            target_degrees = degree(self.target_pyg.edge_index[0])

            # Sort by degree
            source_indices = source_degrees.argsort(descending=True)
            target_indices = target_degrees.argsort(descending=True)

            # Keep top-p nodes
            # source_indices = source_indices[:num_seeds]
            # target_indices = target_indices[:num_seeds]

            seed_dict = {}
            num_added_seeds = 0
            for s_idx in source_indices:
                s_id = self.source_idx2id[s_idx.item()]

                # Check if both source and target have this node
                if s_id in self.target_id2idx.keys():
                    t_idx = self.target_id2idx[s_id]

                    # Check if the target node is in top degree tier
                    # if t_idx in target_indices:
                    #     seed_dict[s_idx.item()] = t_idx
                    seed_dict[s_idx.item()] = t_idx
                    num_added_seeds += 1
                
                if num_added_seeds >= num_seeds:
                    break

        else:
            # Generate global homology mapping of the node indices
            source_indices = []
            target_indices = []
            for s_id, s_idx in self.source_id2idx.items():
                # Target network may doesn't contain the source node
                try:
                    t_idx = self.target_id2idx[s_id]
                    source_indices.append(s_idx)
                    target_indices.append(t_idx)
                except:
                    continue

            # Shuffle and select
            shuff = torch.randperm(len(source_indices))
            shuffled_source_indices = torch.LongTensor(source_indices)[shuff].tolist()
            shuffled_target_indices = torch.LongTensor(target_indices)[shuff].tolist()

            # Select only the `p` percentage of alignments
            selected_source_indices = shuffled_source_indices[:num_seeds]
            selected_target_indices = shuffled_target_indices[:num_seeds]

            # Synthtic alignment dictionary
            seed_dict = dict(zip(selected_source_indices, selected_target_indices))
            
        return seed_dict

    def _random_seed_nodes(self, p=0.2):
        """
        Generate a random mapping between the nodes 
        of source and target networks.

        Args:
            source_id2idx (dict):   Dictionary with mapping from node names 
                                    to node indices in source network.

            target_id2idx (dict):   Dictionary with mapping from node names
                                    to node indices in target network.

            p (float):              Percentage of source nodes to use for 
                                    generate the mapping.

        Returns:
            seed_dict (dict):                   Random node mapping from source to target.
        """

        if self.seed is not None:
            random.seed(self.seed)

        source_indices = list(self.source_id2idx.values())
        target_indices = list(self.target_id2idx.values())

        num_source_nodes_to_map = int(len(source_indices) * p)
        selected_source_indices = random.sample(source_indices, num_source_nodes_to_map)
        selected_target_indices = random.sample(target_indices, num_source_nodes_to_map)

        seed_dict = dict(zip(selected_source_indices, selected_target_indices))

        return seed_dict
    
    def get_pair(self, idx):
        """
        Retun the dictionary with the graph pair informations.
        """

        # Split groundtruth in train, val and test
        if self.gt_mode == 'homology_degree':
            gt_train, gt_val_test = self._generate_homology_degree_gt(top_p=self.train_ratio)
            _, gt_val, gt_test = utils.shuffle_and_split_dict(
                gt_val_test,
                train_ratio=0.0,
                val_ratio=self.val_ratio
            )
            self.gt_dict = utils.combine_dictionaries([gt_train, gt_val, gt_test])

        else:
            gt_train, gt_val, gt_test = utils.shuffle_and_split_dict(
                self.gt_dict,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio
            )

        # Assemble pair informations in a dictionary
        pair_dict = edict()
        pair_dict = {
            'graph_pair': [self.source_pyg, self.target_pyg],
            'id2idx': [self.source_id2idx, self.target_id2idx],
            'idx2id': [self.source_idx2id, self.target_idx2id],
            'gt_full': self.gt_dict,
            'gt_train': gt_train,
            'gt_val': gt_val,
            'gt_test': gt_test
        }
        
        return pair_dict


class SemiSyntheticDataset(Dataset):
    def __init__(self,
                 source_path: str,
                 size: int,
                 permute: Optional[bool] = True,
                 p_add: Optional[float] = 0.0,
                 p_rm: Optional[float] = 0.0,
                 train_ratio: Optional[float] = 0.2,
                 val_ratio: Optional[float] = 0.0,
                 seed: Optional[int] = None):
        
        super(SemiSyntheticDataset, self).__init__()
        
        self.size = size                # Num of target copies
        self.permute = permute          # Choose if permuting the target graph nodes
        self.p_add = p_add              # Prob. of adding an edge
        self.p_rm = p_rm                # Prob. of removing an edge
        self.train_ratio = train_ratio  # Percentage of nodes to use for training
        self.val_ratio = val_ratio      # Percentage of nodes to use for validation
        self.seed = seed                # Random seed

        # Load source graph
        self.source_graph, self.source_id2idx = utils.edgelist_to_pyg(source_path)
        self.source_idx2id = {idx: id for id, idx in self.source_id2idx.items()}

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.get_pair(idx)

    def get_pair(self, idx):
        """
        Generate a random semi-synthetic copy of the source input graph
        and return a dictionary with the source-target graph pair in
        pytorch geometric data format and all the correspoinding informations.
        """
        
        # Generate random semi-synthetic pyg target graph
        self.target_graph, gt_dict = utils.generate_target_graph(
            self.source_graph,
            p_add=self.p_add,
            p_rm=self.p_rm,
            permute=self.permute
        )
        self.target_id2idx = {id: gt_dict[idx] for id, idx in self.source_id2idx.items()}
        self.target_idx2id = {gt_dict[idx]: id  for id, idx in self.source_id2idx.items()}

        # Split alignments in train and test subsets
        gt_train, gt_val, gt_test = utils.shuffle_and_split_dict(
            gt_dict,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio
        )

        # Assemble pair informations in a dictionary
        pair_dict = edict()
        pair_dict = {
            'graph_pair': [self.source_graph, self.target_graph],
            'id2idx': [self.source_id2idx, self.target_id2idx],
            'idx2id': [self.source_idx2id, self.target_idx2id],
            'gt_full': gt_dict,
            'gt_train': gt_train,
            'gt_val': gt_val,
            'gt_test': gt_test
        }
        
        return pair_dict