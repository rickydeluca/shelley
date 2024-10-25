import numpy as np


def dict_to_perm_mat(gt_dict, n_sources, n_targets):
    gt_mat = np.zeros((n_sources, n_targets))
    
    for s, t in gt_dict.items():
        gt_mat[s, t] = 1  
    
    return gt_mat
