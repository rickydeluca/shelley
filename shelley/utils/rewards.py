import torch
from torch_geometric.utils import to_dense_adj


def reward_general(n1, n2, e1, e2, theta_in, miss_match_value=-1.0):
    e1_dense = to_dense_adj(e1, max_num_nodes=n1)
    e2_dense = to_dense_adj(e2, max_num_nodes=n2)

    theta = theta_in[:, :n1, :n2]

    r = e1_dense @ theta @ e2_dense.transpose(-1, -2) * theta
    r = r.sum([1, 2]).mean()

    if miss_match_value > 0.0:
        ones_1 = torch.ones([1, n1, 1]).to(theta_in.device)
        ones_2 = torch.ones([1, n2, 1]).to(theta_in.device)

        r_mis_1 = e1_dense @ ones_1 @ ones_2.transpose(-1, -2) * theta
        r_mis_1 = r_mis_1.sum([1, 2]).mean()

        r_mis_2 = ones_1 @ ones_2.transpose(-1, -2) @ e2_dense.transpose(-1, -2) * theta
        r_mis_2 = r_mis_2.sum([1, 2]).mean()

        r_mis = r_mis_1 + r_mis_2 - 2.0*r

        r_mis = r_mis * miss_match_value

        r = r - r_mis

    return r