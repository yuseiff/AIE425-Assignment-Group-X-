import numpy as np

class MeanCentered_Cosine_similarity_RS:
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=float)
        self.matrix_centered = None
        self.means = None
        self.norms = None
        self.sim_cache = {}

    def fit(self):
        """
        1. Calculate User Means (ignoring NaNs).
        2. Create Mean-Centered Matrix (Rating - Mean).
        3. Pre-compute Norms of Centered Vectors.
        """
        with np.errstate(invalid='ignore'):
            self.means = np.nanmean(self.matrix, axis=1)
            
        self.means = np.nan_to_num(self.means)

        
        self.matrix_centered = self.matrix - self.means[:, np.newaxis]
        
        self.matrix_centered_zeroed = np.nan_to_num(self.matrix_centered)
        
        self.norms = np.sqrt(np.sum(self.matrix_centered_zeroed**2, axis=1))
        self.norms[self.norms == 0] = 1e-9 
        
        print(f"Model fitted. Means & Centered Matrix computed for {len(self.means)} users.")

    def compute_similarities(self, u_idx):
        """
        Calculates Mean-Centered Cosine Similarity between User u_idx and ALL other users.
        """
        if u_idx in self.sim_cache:
            return self.sim_cache[u_idx]
            
        if self.matrix_centered is None: raise Exception("Run .fit() first!")
            
        target_vec = self.matrix_centered_zeroed[u_idx]
        
        dot_products = self.matrix_centered_zeroed.dot(target_vec)
        
        sim_scores = dot_products / (self.norms * self.norms[u_idx])
        
        sim_scores[u_idx] = 0.0
        
        self.sim_cache[u_idx] = sim_scores
        return sim_scores

    def pred(self, u_idx, i_idx, k):
        """
        Prediction using Mean-Centered Formula:
        Pred = Mean_u + (Sum(Sim * (R_vi - Mean_v)) / Sum(|Sim|))
        """
        if self.matrix_centered is None: raise Exception("Run .fit() first!")
        
        sim_scores = self.compute_similarities(u_idx)
        
        item_ratings = self.matrix[:, i_idx]
        
        valid_mask = (~np.isnan(item_ratings)) & (sim_scores > 0)
        
        if not np.any(valid_mask):
            return self.means[u_idx] 
            
        neighbor_sims = sim_scores[valid_mask]
        neighbor_indices = np.where(valid_mask)[0]
        
        if len(neighbor_sims) > k:
            top_k_args = np.argsort(neighbor_sims)[-k:]
            k_sims = neighbor_sims[top_k_args]
            k_indices = neighbor_indices[top_k_args]
        else:
            k_sims = neighbor_sims
            k_indices = neighbor_indices
            
        k_centered_ratings = self.matrix_centered_zeroed[k_indices, i_idx]
        
        numerator = np.sum(k_sims * k_centered_ratings)
        denominator = np.sum(np.abs(k_sims))
        
        if denominator == 0:
            return self.means[u_idx]
            
        pred = self.means[u_idx] + (numerator / denominator)
        
        return np.clip(pred, 0.5, 5.0)
    

import numpy as np

class Discounted_MeanCentered_Cosine_RS(MeanCentered_Cosine_similarity_RS):
    def __init__(self, matrix):
        super().__init__(matrix)
        self.discount_cache = {}

    def _calculate_pair_sim_and_count(self, u_idx, v_idx):
        """
        Calculates Sim and Common Count between User u and User v.
        """
        u_vec = self.matrix[u_idx]
        v_vec = self.matrix[v_idx]
        
        mask = ~np.isnan(u_vec) & ~np.isnan(v_vec)
        count = np.sum(mask)
        
        if count < 2: return 0.0, count
        
        u_centered = u_vec[mask] - self.means[u_idx]
        v_centered = v_vec[mask] - self.means[v_idx]
        
        num = np.dot(u_centered, v_centered)
        den = np.linalg.norm(u_centered) * np.linalg.norm(v_centered)
        
        if den == 0: return 0.0, count
        
        return (num / den), count

    def get_discounted_stats(self, u_idx, beta):
        cache_key = (u_idx, beta)
        if cache_key in self.discount_cache: return self.discount_cache[cache_key]
        
        num_users = self.matrix.shape[0]
        raw_sims = np.zeros(num_users)
        counts = np.zeros(num_users)
        
        for v_idx in range(num_users):
            if u_idx == v_idx: continue
            
            sim, count = self._calculate_pair_sim_and_count(u_idx, v_idx)
            raw_sims[v_idx] = sim
            counts[v_idx] = count
            
        with np.errstate(divide='ignore', invalid='ignore'):
            df = np.minimum(counts, beta) / beta
            
        ds = raw_sims * df
        
        self.discount_cache[cache_key] = (raw_sims, counts, df, ds)
        return raw_sims, counts, df, ds

    def pred_discounted(self, u_idx, i_idx, k, beta):
        _, _, _, ds_scores = self.get_discounted_stats(u_idx, beta)
        
        item_ratings = self.matrix[:, i_idx]
        valid_mask = (~np.isnan(item_ratings)) & (ds_scores > 0)
        
        if not np.any(valid_mask): return self.means[u_idx]
        
        neighbor_sims = ds_scores[valid_mask]
        neighbor_indices = np.where(valid_mask)[0]
        
        if len(neighbor_sims) > k:
            top_k_args = np.argsort(neighbor_sims)[-k:]
            k_sims = neighbor_sims[top_k_args]
            k_indices = neighbor_indices[top_k_args]
        else:
            k_sims = neighbor_sims
            k_indices = neighbor_indices
            
        neighbor_means = self.means[k_indices]
        neighbor_ratings = item_ratings[k_indices]
        deviations = neighbor_ratings - neighbor_means
        
        num = np.sum(k_sims * deviations)
        den = np.sum(np.abs(k_sims))
        
        if den == 0: return self.means[u_idx]
        
        pred = self.means[u_idx] + (num / den)
        return np.clip(pred, 0.5, 5.0)