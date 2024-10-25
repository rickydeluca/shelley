#
# Code from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/permutations.html
#

import torch
import torch.nn as nn


def log_sinkhorn(log_alpha, n_iter, tau):
    """
    Performs incomplete Sinkhorn normalization to log_alpha.

    Args:
      log_alpha: 2D tensor (a matrix of shape [N, N])
        or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
      n_iters: number of sinkhorn iterations (in practice, as little as 20
        iterations are needed to achieve decent convergence for N~100)

    Returns:
      A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
        converted to 3D tensors with batch_size equals to 1)
    """
    
    # Temperature scaling
    log_alpha = log_alpha / tau

    # Sinkhorn iterations
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)

    return log_alpha.exp()


def mask_indices(tensor, row_indices, col_indices):
    batch_size, N, M = tensor.size()
    mask = torch.zeros(batch_size, N, M, dtype=torch.bool, device=tensor.device)
    mask[:, row_indices, :] = True
    mask[:, :, col_indices] = True
    mask[:, :, [i for i in range(M) if i not in col_indices]] = False  # Ensure only specified columns are kept
    mask[:, [i for i in range(M) if i not in row_indices], :] = False  # Ensure only specified rows are kept
    return tensor.masked_fill(~mask, 1e-20)


def log_sinkhorn_norm(log_alpha, n_iter):
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
    return log_alpha.exp()


def gumbel_sinkhorn(log_alpha_in, tau, n_iter, n_sample, noise):
    log_alpha = log_alpha_in.unsqueeze(0).repeat(n_sample, 1, 1)
    if noise:
        uniform_noise = torch.rand_like(log_alpha)
        gumbel_noise = -torch.log(-torch.log(uniform_noise+1e-20)+1e-20)
        log_alpha = (log_alpha + gumbel_noise) / tau
    sampled_perm_mat = log_sinkhorn_norm(log_alpha, n_iter)
    return sampled_perm_mat