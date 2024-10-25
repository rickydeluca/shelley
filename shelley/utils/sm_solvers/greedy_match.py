import torch
import torch_geometric

def greedy_match(S):
    """
    Matches source nodes to target nodes based 
    on a scores matrix using a greedy algorithm, 
    making the operations differentiable using PyTorch.

    Args:
        S (torch.Tensor):
            A scores matrix of shape (MxN) where M 
            is the number of source nodes and N is 
            the number of target nodes.

    Returns:
        torch.Tensor:
            A tensor indicating the matching with 
            shape (MxN), where each element is either 
            0 or 1 indicating the match.
    """
    S = S.T
    m, n = S.shape
    x = S.T.flatten()
    min_size = min([m, n])
    used_rows = torch.zeros(m, dtype=torch.bool)
    used_cols = torch.zeros(n, dtype=torch.bool)
    max_list = torch.zeros(min_size)
    row = torch.zeros(min_size, dtype=torch.long)  # target indexes
    col = torch.zeros(min_size, dtype=torch.long)  # source indexes

    ix = torch.argsort(-x) + 1

    matched = 1
    index = 1
    while matched <= min_size:
        ipos = ix[index - 1]
        jc = torch.ceil(ipos.float() / m).long()
        ic = ipos - (jc - 1) * m
        if ic == 0:
            ic = 1
        if not used_rows[ic - 1] and not used_cols[jc - 1]:
            row[matched - 1] = ic - 1
            col[matched - 1] = jc - 1
            max_list[matched - 1] = x[index - 1]
            used_rows[ic - 1] = True
            used_cols[jc - 1] = True
            matched += 1
        index += 1

    result = torch.zeros(S.T.shape, dtype=torch.float)
    for i in range(len(row)):
        result[col[i], row[i]] = 1
    return result