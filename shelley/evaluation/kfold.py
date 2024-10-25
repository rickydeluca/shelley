import random

import numpy as np
from torch_geometric.loader import DataLoader

from netalign.data.utils import dict_to_perm_mat, move_tensors_to_device
from netalign.evaluation.matchers import greedy_match
from netalign.evaluation.metrics import compute_accuracy
from netalign.models import init_align_model


def shuffle_and_split_in_k(dictionary, k):
    """
    Shuffle the given dictionary and split it into k parts.
    Returns: list(dict)
    """

    items = list(dictionary.items())
    random.shuffle(items)
    
    size = len(items) // k
    remainder = len(items) % k
    
    result = []
    start = 0
    
    for i in range(k):
        end = start + size + (1 if i < remainder else 0)
        part = dict(items[start:end])
        result.append(part)
        start = end
    
    return result


def combine_dictionaries(dict_list):
    """
    Combine a list of dictionary in a single dictionary
    """
    combined_dict = {}
    for d in dict_list:
        combined_dict.update(d)
    return combined_dict


def k_fold_cv(cfg, dataset, k=5, device=None, seed=42):
    model, _ = init_align_model(cfg)
    dataloader = DataLoader(dataset, shuffle=False)
    pair_dict = next(iter(dataloader))

    # Get the total groundtruth
    gt = combine_dictionaries([pair_dict['gt_train'],
                               pair_dict['gt_val'],
                               pair_dict['gt_test']])

    # Shuffle and split it in `k` parts,
    # use 1 for training and `k-1` for testing
    gts_list = shuffle_and_split_in_k(gt, k)

    # Run k-fold
    accs = []
    for i in range(len(gts_list)):
        gt_train = gts_list[i]
        gt_test = combine_dictionaries([gts_list[j] for j in range(len(gts_list)) if j != i])

        pair_dict['gt_train'] = gt_train
        pair_dict['gt_test'] = gt_test

        # Move to device
        pair_dict = move_tensors_to_device(pair_dict, device)
        try:
            model.to(device)
        except: 
            pass

        S, _ = model.align(pair_dict)

        P = greedy_match(S)
        gt_test = dict_to_perm_mat(pair_dict['gt_test'], pair_dict['graph_pair'][0].num_nodes, pair_dict['graph_pair'][1].num_nodes).detach().cpu().numpy()
        acc = compute_accuracy(P, gt_test)

        accs.append(acc)
    
    avg_acc = np.mean(accs)

    return avg_acc