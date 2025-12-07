import numpy as np

def calculate_ari(labels_true, labels_pred):
    """
    Computes Adjusted Rand Index (ARI) without sklearn.
    ARI = (Index - Expected_Index) / (Max_Index - Expected_Index)
    """
    # 1. Contingency Table (Confusion Matrix)
    # Rows = clusters in true, Cols = clusters in pred
    classes_true = np.unique(labels_true)
    classes_pred = np.unique(labels_pred)
    n_true = len(classes_true)
    n_pred = len(classes_pred)
    
    # Map labels to 0..K-1 indices for array indexing
    map_true = {label: i for i, label in enumerate(classes_true)}
    map_pred = {label: i for i, label in enumerate(classes_pred)}
    
    contingency = np.zeros((n_true, n_pred), dtype=np.int64)
    
    for i in range(len(labels_true)):
        r = map_true[labels_true[i]]
        c = map_pred[labels_pred[i]]
        contingency[r, c] += 1
        
    # 2. Components for ARI Formula
    # sum_ij (n_ij choose 2)
    sum_comb_c = np.sum([n * (n - 1) / 2 for n in contingency.flatten()])
    
    # sum_i (a_i choose 2) -> Row sums
    sum_a = np.sum(contingency, axis=1)
    sum_comb_a = np.sum([n * (n - 1) / 2 for n in sum_a])
    
    # sum_j (b_j choose 2) -> Col sums
    sum_b = np.sum(contingency, axis=0)
    sum_comb_b = np.sum([n * (n - 1) / 2 for n in sum_b])
    
    # Total combinations (N choose 2)
    n_samples = len(labels_true)
    total_comb = n_samples * (n_samples - 1) / 2
    
    # 3. ARI Formula
    # Index = sum_comb_c
    # Expected Index = (sum_comb_a * sum_comb_b) / total_comb
    # Max Index = (sum_comb_a + sum_comb_b) / 2
    
    expected_index = (sum_comb_a * sum_comb_b) / total_comb
    max_index = (sum_comb_a + sum_comb_b) / 2
    
    if max_index == expected_index:
        return 0.0 
        
    ari = (sum_comb_c - expected_index) / (max_index - expected_index)
    return ari