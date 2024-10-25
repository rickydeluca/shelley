import networkx as nx
import numpy as np
import torch
from scipy.spatial.distance import cosine
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_networkx
import torch.nn.functional as F


def compute_accuracy(pred, gt):
    n_matched = 0
    for i in range(pred.shape[0]):
        if pred[i].sum() > 0 and np.array_equal(pred[i], gt[i]):
            n_matched += 1
    n_nodes = (gt==1).sum()
    return n_matched/n_nodes


def compute_structural_score(pyg_graph):
    """
    Compute the average structural similarity score of
    the node embeddings of a graph.
    
    Args:
        pyg_graph (Data): A PyTorch Geometric Data object containing node embeddings and edge information.
    
    Returns:
        float: The average structural score of the graph.
    """
    # Extract node embeddings
    embeddings = pyg_graph.x.numpy()
    N = embeddings.shape[0]
    
    # Computing cosine similarities
    cosine_similarities = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            if u != v:
                cosine_similarities[u, v] = 1 - cosine(embeddings[u], embeddings[v])

    # Compute degree differences
    degrees = degree(pyg_graph.edge_index[0]).numpy()
    degree_differences = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            if u != v:
                degree_differences[u, v] = abs(degrees[u] - degrees[v])

    # Compute structural scores
    scores = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            if u != v:
                scores[u, v] = cosine_similarities[u, v] / (1 + degree_differences[u, v])

    # Average the result
    total_score = np.sum(scores) / 2
    average_score = total_score / (N * (N - 1) / 2)

    return average_score


def compute_positional_score(pyg_graph):
    """
    Compute the average positional similarity score of
    the node embeddings of a graph.
    
    Args:
        pyg_graph (Data): A PyTorch Geometric Data object containing node embeddings and edge information.
    
    Returns:
        float: The average positional score of the graph.
    """
    # Extract node embeddings
    embeddings = pyg_graph.x.numpy()
    N = embeddings.shape[0]
    
    # Computing cosine similarities
    cosine_similarities = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            if u != v:
                cosine_similarities[u, v] = 1 - cosine(embeddings[u], embeddings[v])
    
    # Convert to NetworkX to compute shortest paths
    G = to_networkx(pyg_graph)
    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    
    # Compute positional scores
    positional_scores = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            if u != v:
                path_len = shortest_path_lengths[u][v] if v in shortest_path_lengths[u] else float('inf')
                if path_len > 0 and path_len != float('inf'):
                    positional_scores[u, v] = cosine_similarities[u, v] / path_len
                else:
                    positional_scores[u, v] = 0.0
    
    # Get average positional score
    total_positional_score = np.sum(positional_scores) / 2
    average_positional_score = total_positional_score / (N * (N - 1) / 2)

    return average_positional_score


def compute_scores_for_graphs(data_list):
    """
    Compute structural and positional scores for a list of graphs.
    
    Args:
        data_list (list of Data):
            A list of PyTorch Geometric Data objects, each representing a graph.
    
    Returns:
        list(tuple): 
            A list of tuples where each tuple contains the structural 
            and the positional scores of a different graph.
            [(struct1, pos1), (struct2, pos2), ...]

    """
    graph_scores = []
    for data in data_list:
        struct_score = compute_structural_score(data)
        pos_score = compute_positional_score(data)
        graph_scores.append((struct_score, pos_score))
    return graph_scores


def compute_fpsd_score(sim_mat, gt_mat, pred_mat):
    """
    Computes the average absolute distance between the similarity
    scores of the true alignments and the similarity scores of the 
    predicted alignments.

    Args:
        sim_mat (np.ndarray):
            A 2D array of shape (N, M) where each cell (i, j) represents 
            a similarity score between the i-th row element and the j-th column element.
        gt_mat (np.ndarray): 
            A 2D binary array of shape (N, M) where 1 represents the true positive 
            alignments and 0 represents the true negatives. Can be incomplete.
        pred_mat (np.ndarray): 
            A 2D binary array of shape (N, M) where 1 represents the predicted positive 
            alignments and 0 represents the predicted negatives.

    Returns:
        float: The average similartity proximity score
    """
    abs_dists = []
    
    # Iterate over each row of gt_mat
    for i in range(gt_mat.shape[0]):
        # Find columns where there is a true alignment (1) in gt_mat
        true_indices = np.where(gt_mat[i] == 1)[0]
        sim_mean = np.mean(sim_mat[i])
        sim_std = np.std(sim_mat[i])

        for j in true_indices:
            # Compute Z-score distances between sim_mat[i, j] and sim_mat[i, k] for k in pred_indices
            pred_indices = np.where(pred_mat[i] == 1)[0]
            abs_dists.extend((np.abs(sim_mat[i, j] - sim_mat[i, pred_indices]) - sim_mean) / sim_std)

    # Compute the average absolute distance
    avg_dist = np.mean(abs_dists)
    
    return avg_dist



def compute_fpsd(sim_mat, gt_perm, pred_perm):
    """
    Calculate the False Positive Similarity Difference (FPSD) Metric.

    This metric computes the percentage difference between the average similarity scores
    of false positive alignments and the average similarity scores of true positive alignments.

    Args:
        sim_mat (list or ndarray): Similarity matrix where each cell (i,j) represents a similarity measure
                                   between the embedding of node_i from the first network and node_j from the second network.
        gt_perm (list or ndarray): Ground truth permutation matrix indicating the real alignments between nodes of the two networks.
        pred_perm (list or ndarray): Predicted permutation matrix indicating the predicted alignments by the model.

    Returns:
        float: The percentage difference in similarity scores between false positives and true positives.
    """
    # Ensure the matrices are in torch tensors
    sim_mat = torch.tensor(sim_mat, dtype=torch.float32)
    gt_perm = torch.tensor(gt_perm, dtype=torch.float32)
    pred_perm = torch.tensor(pred_perm, dtype=torch.float32)
    
    # Identify true positives and false positives
    true_positives = gt_perm * pred_perm
    false_positives = (1 - gt_perm) * pred_perm
    
    # Extract similarity scores for true positives and false positives
    true_positive_scores = sim_mat[true_positives == 1]
    false_positive_scores = sim_mat[false_positives == 1]
    
    # Ensure there are no empty tensors
    if true_positive_scores.numel() == 0 or false_positive_scores.numel() == 0:
        raise ValueError("There are no true positives or false positives in the provided matrices.")
    
    # Calculate the average similarity score of true positives
    avg_true_positive_score = true_positive_scores.mean().item()
    
    # Calculate the average similarity score of false positives
    avg_false_positive_score = false_positive_scores.mean().item()
    
    # Calculate the percentage difference
    percentage_difference = abs(avg_true_positive_score - avg_false_positive_score) / avg_true_positive_score
    
    return percentage_difference

def compute_conf_score(sim_mat):
    """
    Compute the average confidence of the model 
    as the average distance between the higher
    similarity score in each row and the other scores.
    """

    distances = []
    for _row in sim_mat:
        # Compute percentages
        row = torch.from_numpy(_row).to(torch.float)
        row_perc = F.softmax(row, dim=0)
        
        # Find the top 2 values and their indices
        values, indices = torch.topk(row_perc, k=2)
        
        # Extract max and second_max values
        max_val = values[0]
        second_max_val = values[1]
        
        # Compute the absolute difference
        difference = max_val - second_max_val

        distances.append(difference.item())

    return np.mean(distances)


if __name__ == '__main__':

    # Test structural and positional scores:
    node_features_1 = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.float)
    edge_index_1 = torch.tensor([[0, 1, 2, 3, 0], [1, 0, 3, 2, 2]], dtype=torch.long)
    data_1 = Data(x=node_features_1, edge_index=edge_index_1)

    node_features_2 = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=torch.float)
    edge_index_2 = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    data_2 = Data(x=node_features_2, edge_index=edge_index_2)

    data_list = [data_1, data_2]

    graph_scores = compute_scores_for_graphs(data_list)
    print("Graph Scores (struc, pos):", graph_scores)

    # Test similarity proximity score
    sim_mat = np.array([[0.1, 0.4, 0.3], [0.7, 0.5, 0.2], [0.6, 0.9, 0.8]])
    gt_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    pred_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])

    fpsd = compute_fpsd(sim_mat, gt_mat, pred_mat)
    print("Similarity Proximity Score:", fpsd)

    # Test confidence score
    sim_mat = np.array([[1, -1, -1], [1, 2, 3], [1, 10, 100]])
    cs = compute_conf_score(sim_mat)
    print("Confidence Score:", cs)
