import numpy as np
import torch

def stable_marriage(ranking_matrix):
    """
    Compute a permutation matrix using the stable marriage algorithms
    on a ranking matrix.
    """
    n = ranking_matrix.size(0)

    # Convert ranking matrix to preference lists
    men_preferences = ranking_matrix.argsort(dim=1, descending=True)
    women_preferences = ranking_matrix.argsort(dim=0, descending=True)

    # Gale-Shapley algorithm to find stable matching
    free_men = list(range(n))
    women_partner = [-1] * n
    men_next_proposal = [0] * n
    women_preferences_rank = torch.zeros((n, n), dtype=torch.int64)

    for i in range(n):
        for j in range(n):
            women_preferences_rank[i, women_preferences[j, i]] = j

    while free_men:
        man = free_men.pop(0)
        woman = men_preferences[man, men_next_proposal[man]]
        men_next_proposal[man] += 1

        if women_partner[woman] == -1:
            women_partner[woman] = man
        else:
            current_partner = women_partner[woman]
            if women_preferences_rank[woman, man] < women_preferences_rank[woman, current_partner]:
                women_partner[woman] = man
                free_men.append(current_partner)
            else:
                free_men.append(man)

    permutation_matrix = torch.zeros((n, n))
    for woman, man in enumerate(women_partner):
        permutation_matrix[man, woman] = 1

    return permutation_matrix

def greedy_match(S):
    """
    Matches source nodes to target nodes based 
    on a scores matrix using a greedy algorithm.

    Args:
        S (numpy.ndarray):
            A scores matrix of shape (MxN) where M 
            is the number of source nodes and N is 
            the number of target nodes.

    Returns:
        dict:
            A dictionary mapping each source node 
            to a list of target nodes.
    """
    S = S.T
    m, n = S.shape
    x = S.T.flatten()
    min_size = min([m,n])
    used_rows = np.zeros((m))
    used_cols = np.zeros((n))
    max_list = np.zeros((min_size))
    row = np.zeros((min_size))  # target indexes
    col = np.zeros((min_size))  # source indexes

    ix = np.argsort(-x) + 1

    matched = 1
    index = 1
    while(matched <= min_size):
        ipos = ix[index-1]
        jc = int(np.ceil(ipos/m))
        ic = ipos - (jc-1)*m
        if ic == 0: ic = 1
        if (used_rows[ic-1] == 0 and used_cols[jc-1] == 0):
            row[matched-1] = ic - 1
            col[matched-1] = jc - 1
            max_list[matched-1] = x[index-1]
            used_rows[ic-1] = 1
            used_cols[jc-1] = 1
            matched += 1
        index += 1

    result = np.zeros(S.T.shape)
    for i in range(len(row)):
        result[int(col[i]), int(row[i])] = 1
    return result