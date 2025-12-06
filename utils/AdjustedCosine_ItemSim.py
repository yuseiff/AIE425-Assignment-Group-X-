import numpy as np

class AdjustedCosine_ItemBased_RS:
    def __init__(self, item_user_matrix):
        self.matrix = np.array(item_user_matrix, dtype=float)
        self.user_means = None
        self.matrix_centered = None
        self.norms = None
        self.sim_cache = {}

    def fit(self):
        """
        1. Calculate User Means (Column Means).
        2. Center the matrix by subtracting User Means.
        3. Pre-compute Norms for Item Vectors.
        """
        with np.errstate(invalid='ignore'):
            self.user_means = np.nanmean(self.matrix, axis=0)
            
        self.user_means = np.nan_to_num(self.user_means)
        self.matrix_centered = self.matrix - self.user_means
        
        self.matrix_centered_zeroed = np.nan_to_num(self.matrix_centered)
        self.norms = np.sqrt(np.sum(self.matrix_centered_zeroed**2, axis=1))
        self.norms[self.norms == 0] = 1e-9
        
        print(f"Item-Based Model fitted. Matrix Shape: {self.matrix.shape}")

    def compute_similarities(self, item_idx):
        """
        Calculates Adjusted Cosine Similarity between 'item_idx' and all other items.
        """
        if item_idx in self.sim_cache:
            return self.sim_cache[item_idx]
            
        if self.matrix_centered is None: raise Exception("Run .fit() first!")
            
        target_vec = self.matrix_centered_zeroed[item_idx]
        
        dot_products = self.matrix_centered_zeroed.dot(target_vec)
        
        sim_scores = dot_products / (self.norms * self.norms[item_idx])
        
        sim_scores[item_idx] = 0.0
        
        self.sim_cache[item_idx] = sim_scores
        return sim_scores

    def pred(self, u_idx, i_idx, k):
        """
        Item-Based Prediction:
        Pred(u, i) = weighted average of ratings user 'u' gave to items similar to 'i'.
        """
        
        sim_scores = self.compute_similarities(i_idx)
        
        user_ratings = self.matrix[:, u_idx]
        
        valid_mask = (~np.isnan(user_ratings)) & (sim_scores > 0)
        
        if not np.any(valid_mask): return 0.0 
            
        neighbor_sims = sim_scores[valid_mask]
        neighbor_ratings = user_ratings[valid_mask]
        
        if len(neighbor_sims) > k:
            top_k_args = np.argsort(neighbor_sims)[-k:]
            k_sims = neighbor_sims[top_k_args]
            k_ratings = neighbor_ratings[top_k_args]
        else:
            k_sims = neighbor_sims
            k_ratings = neighbor_ratings
            
        if np.sum(k_sims) == 0: return 0.0
        
        pred = np.sum(k_sims * k_ratings) / np.sum(k_sims)
        return pred
    
class Discounted_AdjustedCosine_ItemBased_RS(AdjustedCosine_ItemBased_RS):
    """
    Extends Item-Based RS to add Discounting logic.
    """
    def __init__(self, item_user_matrix):
        super().__init__(item_user_matrix)
        self.discount_cache = {}

    def _calculate_pair_sim_and_count(self, i_idx, j_idx):
        """
        Calculates Adj Cosine AND Common User Count between Item i and Item j.
        """
        if self.matrix_centered_zeroed is None: raise Exception("Run .fit() first!")
        
        vec_i = self.matrix_centered_zeroed[i_idx]
        vec_j = self.matrix_centered_zeroed[j_idx]
        
        raw_i = self.matrix[i_idx]
        raw_j = self.matrix[j_idx]
        mask = ~np.isnan(raw_i) & ~np.isnan(raw_j)
        count = np.sum(mask)
        
        if count == 0: return 0.0, 0
            
        
        dot = np.dot(vec_i, vec_j)
        norm_i = self.norms[i_idx]
        norm_j = self.norms[j_idx]
        
        if norm_i == 0 or norm_j == 0: return 0.0, count
            
        sim = dot / (norm_i * norm_j)
        return sim, count

    def get_discounted_stats(self, i_idx, beta):
        """
        Returns: (Raw_Sim, Common_Counts, DF, DS) for target Item i against all items.
        """
        cache_key = (i_idx, beta)
        if cache_key in self.discount_cache: return self.discount_cache[cache_key]

        num_items = self.matrix.shape[0]
        raw_sims = np.zeros(num_items)
        counts = np.zeros(num_items)
        
        for j_idx in range(num_items):
            if i_idx == j_idx: continue
            
            sim, count = self._calculate_pair_sim_and_count(i_idx, j_idx)
            raw_sims[j_idx] = sim
            counts[j_idx] = count
            
        with np.errstate(divide='ignore', invalid='ignore'):
            df = np.minimum(counts, beta) / beta
            
        ds = raw_sims * df
        
        self.discount_cache[cache_key] = (raw_sims, counts, df, ds)
        return raw_sims, counts, df, ds

    def pred_discounted(self, u_idx, i_idx, k, beta):
        """
        Item-Based Prediction using Discounted Similarity.
        """
        _, _, _, ds_scores = self.get_discounted_stats(i_idx, beta)
        
        user_ratings = self.matrix[:, u_idx]
        
        valid_mask = (~np.isnan(user_ratings)) & (ds_scores > 0)
        
        if not np.any(valid_mask): return 0.0
            
        neighbor_sims = ds_scores[valid_mask]
        neighbor_ratings = user_ratings[valid_mask]
        
        if len(neighbor_sims) > k:
            top_k_args = np.argsort(neighbor_sims)[-k:]
            k_sims = neighbor_sims[top_k_args]
            k_ratings = neighbor_ratings[top_k_args]
        else:
            k_sims = neighbor_sims
            k_ratings = neighbor_ratings
            
        if np.sum(k_sims) == 0: return 0.0
        
        return np.sum(k_sims * k_ratings) / np.sum(k_sims)