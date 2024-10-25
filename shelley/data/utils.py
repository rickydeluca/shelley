import json
import os
import random
from itertools import chain

import networkx as nx
import numpy as np
import torch
from networkx.readwrite import json_graph
from torch_geometric.data import Data, Batch
from torch_geometric.utils import (add_random_edge, contains_self_loops,
                                   dropout_edge, is_undirected,
                                   remove_self_loops)
from torch_geometric.utils.convert import from_networkx


def network_info(G, n=None):
    """Print short summary of information for the graph G or the node n.

    Args:
        G: A NetworkX graph.
        n:  A node in the graph G
    """
    info='' # Append this all to a string
    if n is None:
        info+="Name: %s\n"%G.name
        type_name = [type(G).__name__]
        info+="Type: %s\n"%",".join(type_name)
        info+="Number of nodes: %d\n"%G.number_of_nodes()
        info+="Number of edges: %d\n"%G.number_of_edges()
        nnodes=G.number_of_nodes()
        if len(G) > 0:
            if G.is_directed():
                info+="Average in degree: %8.4f\n"%\
                    (sum(G.in_degree().values())/float(nnodes))
                info+="Average out degree: %8.4f"%\
                    (sum(G.out_degree().values())/float(nnodes))
            else:
                degrees = dict(G.degree())
                s=sum(degrees.values())
                info+="Average degree: %8.4f"%\
                    (float(s)/float(nnodes))

    else:
        if n not in G:
            raise nx.NetworkXError("node %s not in graph"%(n,))
        info+="Node % s has the following properties:\n"%n
        info+="Degree: %d\n"%G.degree(n)
        info+="Neighbors: "
        info+=' '.join(str(nbr) for nbr in G.neighbors(n))

    return info


def edgelist_to_graphsage(dir, seed=42):
    np.random.seed(seed)
    edgelist_path = dir + "/edgelist/edgelist"

    # Check if the edge list is weighted or unweighted
    with open(edgelist_path, 'r') as file:
        first_line = file.readline().strip().split()
        is_weighted = len(first_line) > 2

    if is_weighted:
        G = nx.read_weighted_edgelist(edgelist_path)
    else:
        G = nx.read_edgelist(edgelist_path)

    print(network_info(G))

    id2idx = {}
    for i, node in enumerate(G.nodes()):
        id2idx[str(node)] = i

    res = json_graph.node_link_data(G)
    res['nodes'] = [{'id': node['id']} for node in res['nodes']]

    if is_weighted:
        res['links'] = [{'source': link['source'],
                         'target': link['target'],
                         'weight': link['weight']}
                         for link in res['links']]
    else:
        res['links'] = [{'source': link['source'],
                         'target': link['target']}
                         for link in res['links']] 

    if not os.path.exists(dir + "/graphsage/"):
        os.makedirs(dir + "/graphsage/")

    with open(dir + "/graphsage/" + "G.json", 'w') as outfile:
        json.dump(res, outfile)
    with open(dir + "/graphsage/" + "id2idx.json", 'w') as outfile:
        json.dump(id2idx, outfile)


def edgelist_to_networkx(edgelist_path, verbose=False):
    """
    Read the edgelist file in the `dir` path, check if it
    is weighted and return the corresponding NetworkX graph.
    """

    # Check if the edge list is weighted or unweighted
    with open(edgelist_path, 'r') as file:
        first_line = file.readline().strip().split()
        is_weighted = len(first_line) > 2

    if is_weighted:
        G = nx.read_weighted_edgelist(edgelist_path)
    else:
        G = nx.read_edgelist(edgelist_path)
        nx.set_edge_attributes(G, float(1), name='weight') # Explicit weight value

    if verbose:
        print(network_info(G))

    return G


def get_node_attribute_names(G):
    """
    Return the list with the name of node attributes
    of the NetworkX graph `G`.

    If `G` has no node attributes return `None`.
    """
    attrs_list = set(chain.from_iterable(d.keys() for *_, d in G.nodes(data=True)))

    if len(attrs_list) > 0:
        return attrs_list
    else:
        return None


def get_edge_attribute_names(G):
    """
    Return the list with the name of edge attributes
    of the NetworkX graph `G`.

    If `G` has no edge attributes return `None`.
    """
    attrs_list = set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True)))
    
    if len(attrs_list) > 0:
        return attrs_list
    else:
        return None


def edgelist_to_pyg(edgelist_path, verbose=False):
    """
    Converts a graph from an edge list in the specified directory to a PyTorch Geometric (PyG) graph.

    Args:
        dir (str): The directory containing the edge list file.
        verbose (bool, optional): If True print network infos. Default: False.

    Returns:
        torch_geometric.data.Data: PyTorch Geometric graph object representing the loaded graph.
    """
    # Load graph in NetworkX format
    G = edgelist_to_networkx(edgelist_path, verbose=verbose)
    id2idx = {id: idx for idx, id in enumerate(G.nodes())}
    
    # Get list of node and edge attributes
    node_attrs_list = get_node_attribute_names(G)
    edge_attrs_list = get_edge_attribute_names(G)
    
    # Convert to PyG
    pyg_graph = from_networkx(
        G,
        group_node_attrs=node_attrs_list,
        group_edge_attrs=edge_attrs_list
    )

    # Add additional attributes
    pyg_graph.num_nodes = G.number_of_nodes()

    # Remove self loops
    if contains_self_loops(pyg_graph.edge_index):
        pyg_graph.edge_index, pyg_graph.edge_attr = remove_self_loops(pyg_graph.edge_index,
                                                                      pyg_graph.edge_attr)
    return pyg_graph, id2idx


def permute_graph(pyg_source):
    """
    Permute the indices of the PyTorch Geometric 
    Data object `pyg_source` and return the permuted
    copy along with the mapping of node indices.
    """
    # Clone graph
    pyg_target = pyg_source.clone()

    # Generate node mapping
    num_nodes = pyg_target.num_nodes
    perm = torch.randperm(num_nodes)
    mapping = {i: perm[i].item() for i in range(num_nodes)}

    # Permute graph
    pyg_target.edge_index = perm[pyg_target.edge_index]

    return pyg_target, mapping


def sample_edge_attrs(edge_attr, num_new_attrs=0):
    """
    Sample `num_new_attrs` with the same size L of the attributes
    in `edge_attr` and value between the minimum and the maximum
    value in the `edge_attrs` tensor.

    Returns:
        torch.Tensor:   The sample new attributes of shape (num_new_attrs, L)
    """
    # Calculate min and max values along each column
    min_values, _ = torch.min(edge_attr, dim=1, keepdim=True)
    max_values, _ = torch.max(edge_attr, dim=1, keepdim=True)

    # Create a new tensor with values sampled between min and max
    sampled_attrs = torch.rand_like(edge_attr)
    sampled_attrs = sampled_attrs * (max_values - min_values) + min_values

    return sampled_attrs[:num_new_attrs, :]


def remove_random_edges(pyg_graph, p=0.0):
    """
    Drop random edges from `pyg_graph`. The probability
    of one edge to be removed is given by `p`.
    Drop also the attributes corresponding to the dropped edges.

    Returns:
        torch_geometric.data.Data:  The pyg graph with dropped edges.
    """

    edge_index = pyg_graph.edge_index
    edge_attr = pyg_graph.edge_attr

    # Check if undirected
    if is_undirected(edge_index, edge_attr):
        force_undirected = True
    else:
        force_undirected = False

    # Remove edges
    new_edge_index, edge_mask = dropout_edge(edge_index, p,
                                             force_undirected=force_undirected)
    new_edge_attr = edge_attr[edge_mask]

    # Return new graph
    pyg_graph.edge_index = new_edge_index
    pyg_graph.edge_attr = new_edge_attr

    return pyg_graph


def add_random_edges(pyg_graph, p=0.0):
    """
    Add random edges to `pyg_graph`. The percentage of new edges 
    is given by `p` and it is computed on the basis of the already
    existing edges.

    Also, generate a new node attribute for each new added edge.
    The new node attribute is sampled between the min and max 
    actual attribute values. If they are all the same a new attribute
    with the same value of those already present is sampled.

    Returns:
        torch_geometric.data.Data:  The pyg graph with new added edges.
    """

    edge_index = pyg_graph.edge_index
    edge_attr = pyg_graph.edge_attr

    # Check if undirected
    if is_undirected(edge_index, edge_attr):
        force_undirected = True
    else:
        force_undirected = False

    # Add edges
    new_edge_index, added_edges = add_random_edge(edge_index, p,
                                                  force_undirected=force_undirected)
    
    # Sample edge attributes for the new added edges
    sampled_edge_attr = sample_edge_attrs(edge_attr, num_new_attrs=added_edges.size(1))

    # Concat new samples attributes to the original attributes
    new_edge_attr = torch.cat((edge_attr, sampled_edge_attr), dim=0)

    # Return the new graph
    pyg_graph.edge_index = new_edge_index
    pyg_graph.edge_attr = new_edge_attr

    return pyg_graph


def permute_and_modify_graph(G, permute=True, p_add=0.0, p_rm=0.0):
    # Create a copy of the original graph
    G_permuted = G.copy()
    
    # Get the list of nodes and permute them
    nodes = list(G.nodes())
    permuted_nodes = nodes.copy()
    if permute:
        random.shuffle(permuted_nodes)
    
    # Create a mapping from original nodes to permuted nodes
    mapping = {original: permuted for original, permuted in zip(nodes, permuted_nodes)}
    
    # Relabel the nodes of the graph according to the permuted mapping
    G_permuted = nx.relabel_nodes(G_permuted, mapping)
    
    # Add dummy edges with probability p_add
    num_edges_to_add = int(p_add * len(G.number_of_edges()))
    
    added_edges = set()
    while len(added_edges) < num_edges_to_add:
        u, v = random.sample(nodes, 2)
        if not G_permuted.has_edge(u, v):
            G_permuted.add_edge(u, v)
            added_edges.add((u, v))
    
    # Remove edges with probability p_rm
    edges_to_remove = []
    for u, v in G_permuted.edges():
        if random.random() < p_rm:
            edges_to_remove.append((u, v))
    
    G_permuted.remove_edges_from(edges_to_remove)
    
    return G_permuted, mapping
    

def generate_target_graph(pyg_source, p_rm=0.0, p_add=0.0, permute=True):
    """
    Generate the permuted and noised target graph obtained from
    the pytorch geometric Data object `pyg_source`.
    
    Args:
        pyg_source (torch_geometric.data.Data):     The pytorch geometric input graph.
        p_rm (float):                               The probability of dropping an existing edge.
        p_add (float):                              The probabilty of adding a new edge.

    Returns:
        pyg_target (torch_geometric.data.Data):     The radomly permuted and noised target graph.
        mapping (dict):                             The groundtruth mapping of node indices.
    """
    
    if permute:
        pyg_target, mapping = permute_graph(pyg_source)
    else:
        pyg_target = pyg_source.clone()
        mapping = {i: i for i in range(pyg_source.num_nodes)}

    # Remove and/or add edges with probability
    pyg_target = remove_random_edges(pyg_target, p=p_rm)
    pyg_target = add_random_edges(pyg_target, p=p_add)

    # Remove any evantual self loop
    if contains_self_loops(pyg_target.edge_index):
        pyg_target.edge_index, pyg_target.edge_attr = remove_self_loops(pyg_target.edge_index,
                                                                        pyg_target.edge_attr)

    return pyg_target, mapping


def shuffle_and_split_dict(dictionary, train_ratio=0.0, val_ratio=0.0):
    """
    Shuffle the items in the dictionary and split it based on the given ratios.

    Args:
        dictionary (dict): The input dictionary to shuffle and split.
        train_ratio (float, optional): The ratio for the training set.
        val_ratio (float, optional): The ratio for the validation set.

    Returns:
        dict, dict, dict: Three dictionaries representing the training, validation, and test groundtruths.
    """

    # Shuffle dictionary items
    items = list(dictionary.items())
    random.shuffle(items)

    # Split dictionary
    train_index = int(len(items) * train_ratio)
    val_index = int(len(items) * (train_ratio + val_ratio))
    val_items = items[train_index:val_index]

    train_items = items[:train_index]
    val_items = items[train_index:val_index]
    test_items = items[val_index:]
    
    train_dict = {k: v for k, v in train_items}
    val_dict = {k: v for k, v in val_items}
    test_dict = {k: v for k, v in test_items}

    return train_dict, val_dict, test_dict


def read_dict(path, id2idx_s: dict, id2idx_t: dict):
    with open(path, 'r') as file:
        my_dict = {}
        for line in file:
            nodes = line.split()
            src = id2idx_s[nodes[0]]
            tgt = id2idx_t[nodes[1]]

            # Add the key-value pair to the dictionary
            my_dict[src] = tgt
    
    return my_dict


def create_alignment_matrix(id2idx_s, id2idx_t):
    num_nodes_s = len(id2idx_s)
    num_nodes_t = len(id2idx_t)

    alignment_matrix = torch.zeros(num_nodes_s, num_nodes_t, dtype=torch.bool)

    for id_s, idx_s in id2idx_s.items():
        idx_t = id2idx_t.get(id_s, None)
        if idx_t is not None:
            alignment_matrix[idx_s, idx_t] = 1

    return alignment_matrix


def generate_split_masks(groundtruth_matrix, train_ratio, val_ratio=0):
    # Shuffle and split the indices
    num_samples = groundtruth_matrix.size(0)
    indices = torch.randperm(num_samples)

    num_train = int(train_ratio * num_samples)
    num_val = int(val_ratio * num_samples)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train+num_val]
    test_indices = indices[num_train+num_val:]

    # Create masks
    train_mask = torch.zeros_like(groundtruth_matrix, dtype=torch.long)
    val_mask = torch.zeros_like(groundtruth_matrix, dtype=torch.long)
    test_mask = torch.zeros_like(groundtruth_matrix, dtype=torch.long)

    train_mask[train_indices] = 1
    val_mask[val_indices] = 1
    test_mask[test_indices] = 1

    return train_mask, val_mask, test_mask

def split_groundtruth_matrix(groundtruth_matrix, train_ratio, val_ratio=0):
    """
    Splits the groundtruth matrix into train, validation, and test matrices 
    based on the provided ratios.

    Args:
        groundtruth_matrix (torch.Tensor):
            The original groundtruth matrix.

        train_ratio (float):
            The ratio of samples to be used for training.
            
        val_ratio (float, optional):
            The ratio of samples to be used for validation. 
            Defaults to 0.

    Returns:
        torch.Tensor: Masked train matrix.
        torch.Tensor: Masked validation matrix.
        torch.Tensor: Masked test matrix.
    """
    # Shuffle and split the indices
    num_samples = groundtruth_matrix.size(0)
    indices = torch.randperm(num_samples)

    num_train = int(train_ratio * num_samples)
    num_val = int(val_ratio * num_samples)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train+num_val]
    test_indices = indices[num_train+num_val:]

    # Create masks
    train_mask = torch.zeros_like(groundtruth_matrix, dtype=torch.long)
    val_mask = torch.zeros_like(groundtruth_matrix, dtype=torch.long)
    test_mask = torch.zeros_like(groundtruth_matrix, dtype=torch.long)

    train_mask[train_indices] = 1
    val_mask[val_indices] = 1
    test_mask[test_indices] = 1

    # Apply masks
    masked_train = groundtruth_matrix * train_mask
    masked_val = groundtruth_matrix * val_mask
    masked_test = groundtruth_matrix * test_mask

    return masked_train, masked_val, masked_test

def move_tensors_to_device(data, device):
    """
    Recursively move torch tensors in a dictionary, list, tuple, or PyTorch Geometric Data object to the specified device.

    Parameters:
        data (dict, list, tuple, torch_geometric.data.Data): Input data structure containing elements to be checked and moved.
        device: Device to which tensors should be moved (e.g., 'cpu' or 'cuda').

    Returns:
        dict, list, tuple, or torch_geometric.data.Data: Updated data structure with tensors moved to the specified device.
    """
    if isinstance(data, dict):
        return {key: move_tensors_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_tensors_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_tensors_to_device(item, device) for item in data)
    elif isinstance(data, Data) or isinstance(data, Batch):
        data.x = data.x.to(device) if data.x is not None else None
        data.edge_index = data.edge_index.to(device) if data.edge_index is not None else None
        data.edge_attr = data.edge_attr.to(device) if data.edge_attr is not None else None
        data.batch = data.batch.to(device) if data.batch is not None else None
        data.ptr = data.ptr.to(device) if data.ptr is not None else None
        return data
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data
    

def dict_to_perm_mat(dict, num_source_nodes, num_target_nodes):
    """
    Given a dictionary with keys the nodes of the source network
    and values the corresponding indices of the target network,
    generates the permutation matrix.
    """
    perm_mat = torch.zeros((num_source_nodes, num_target_nodes))
    for s, t in dict.items():
        perm_mat[s, t] = 1
    return perm_mat


def invert_dict(dict):
    """
    Swap key and values of a dictionary.
    """
    return {v: k for k, v in dict.items()}


def replace_tensor_items(dict):
    """
    Given a dictionary, replace any monodimensional 
    tensor keys or values with their corresponding items.
    """    
    new_dict = {}
    
    for key, value in dict.items():
        # Check if the key is a 1D tensor
        if isinstance(key, torch.Tensor) and key.ndimension() == 1 and key.size(0) == 1:
            key = key.item()
        
        # Check if the value is a 1D tensor
        if isinstance(value, torch.Tensor) and value.ndimension() == 1 and value.size(0) == 1:
            value = value.item()
        
        new_dict[key] = value
    
    return new_dict


def get_valid_matrix_mask(matrix):
    """
    If the input matrix contains rows and columns with all zeros,
    produce the mask that keep only the rows/cols with at least
    one non-zero element.
    """
    matrix = torch.tensor([[1,0,0],
                           [0,1,0],
                           [0,0,0]])
    if len(matrix.shape) == 2:
        matrix = matrix.unsqueeze(0)
    elif len(matrix.shape) == 3:
        pass
    else:
        raise ValueError("Invalid matrix shape.")
    
    # Find valid rows/cols
    valid_rows = []
    valid_cols = []
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[2]):
            if matrix[0,i,j] != 0:
                valid_rows.append(i)
                valid_cols.append(j)

    # Produce the mask
    mask = torch.zeros_like(matrix)
    for r in valid_rows:
        mask[:,r,:] = 1
    for c in valid_cols:
        mask[:,:,c] = 1
        
    return mask

def combine_dictionaries(dict_list):
    """
    Combine a list of dictionary in a single dictionary
    """
    combined_dict = {}
    for d in dict_list:
        combined_dict.update(d)
    return combined_dict