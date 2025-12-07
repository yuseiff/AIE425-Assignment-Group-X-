import numpy as np

class Pearson_ItemBased_RS:
    def __init__(self, item_user_matrix):
        # Rows = Items, Cols = Users
        self.matrix = np.array(item_user_matrix, dtype=float)
        self.item_means = None
        self.sim_cache = {}

    def fit(self):
        """
        Calculates Global Item Means (used for Prediction baseline).
        """
        with np.errstate(invalid='ignore'):
            self.item_means = np.nanmean(self.matrix, axis=1)
        self.item_means = np.nan_to_num(self.item_means)
        print(f"Pearson Item-Based Model fitted. Means computed for {len(self.item_means)} items.")

    def _calculate_pair_pearson(self, i_idx, j_idx):
        """
        Calculates Pearson Correlation between Item i and Item j.
        Uses LOCAL MEANS (mean of ratings for just the common users).
        """
        vec_i = self.matrix[i_idx]
        vec_j = self.matrix[j_idx]
        
        mask = ~np.isnan(vec_i) & ~np.isnan(vec_j)
        
        if np.sum(mask) < 2: return 0.0
        
        common_i = vec_i[mask]
        common_j = vec_j[mask]
        
        mean_i = np.mean(common_i)
        mean_j = np.mean(common_j)
        
        cent_i = common_i - mean_i
        cent_j = common_j - mean_j
        
        num = np.dot(cent_i, cent_j)
        den = np.linalg.norm(cent_i) * np.linalg.norm(cent_j)
        
        if den == 0: return 0.0
        return num / den

    def compute_similarities(self, i_idx):
        if i_idx in self.sim_cache: return self.sim_cache[i_idx]
        
        num_items = self.matrix.shape[0]
        sims = np.zeros(num_items)
        
        for j_idx in range(num_items):
            if i_idx == j_idx: continue
            sims[j_idx] = self._calculate_pair_pearson(i_idx, j_idx)
            
        self.sim_cache[i_idx] = sims
        return sims

    def pred(self, u_idx, i_idx, k):
        """
        Prediction: Item Mean + Weighted Deviation of Neighbors
        """
        sim_scores = self.compute_similarities(i_idx)
        
        user_ratings = self.matrix[:, u_idx]
        valid_mask = (~np.isnan(user_ratings)) & (sim_scores > 0)
        
        if not np.any(valid_mask): return self.item_means[i_idx]
        
        neighbor_sims = sim_scores[valid_mask]
        neighbor_indices = np.where(valid_mask)[0]
        
        if len(neighbor_sims) > k:
            top_k_args = np.argsort(neighbor_sims)[-k:]
            k_sims = neighbor_sims[top_k_args]
            k_indices = neighbor_indices[top_k_args]
        else:
            k_sims = neighbor_sims
            k_indices = neighbor_indices
            
        neighbor_means = self.item_means[k_indices]
        neighbor_ratings = user_ratings[k_indices]
        deviations = neighbor_ratings - neighbor_means
        
        num = np.sum(k_sims * deviations)
        den = np.sum(np.abs(k_sims))
        
        if den == 0: return self.item_means[i_idx]
        
        pred = self.item_means[i_idx] + (num / den)
        return np.clip(pred, 0.5, 5.0)

class Discounted_Pearson_ItemBased_RS(Pearson_ItemBased_RS):
    def __init__(self, matrix):
        super().__init__(matrix)
        self.discount_cache = {}

    def _calculate_pair_pearson_and_count(self, i_idx, j_idx):
        vec_i = self.matrix[i_idx]
        vec_j = self.matrix[j_idx]
        mask = ~np.isnan(vec_i) & ~np.isnan(vec_j)
        count = np.sum(mask)
        
        if count < 2: return 0.0, count
        
        common_i = vec_i[mask]
        common_j = vec_j[mask]
        
        cent_i = common_i - np.mean(common_i)
        cent_j = common_j - np.mean(common_j)
        
        num = np.dot(cent_i, cent_j)
        den = np.linalg.norm(cent_i) * np.linalg.norm(cent_j)
        
        if den == 0: return 0.0, count
        return (num / den), count

    def get_discounted_stats(self, i_idx, beta):
        cache_key = (i_idx, beta)
        if cache_key in self.discount_cache: return self.discount_cache[cache_key]
        
        num_items = self.matrix.shape[0]
        raw_sims = np.zeros(num_items)
        counts = np.zeros(num_items)
        
        for j_idx in range(num_items):
            if i_idx == j_idx: continue
            sim, count = self._calculate_pair_pearson_and_count(i_idx, j_idx)
            raw_sims[j_idx] = sim
            counts[j_idx] = count
            
        with np.errstate(divide='ignore', invalid='ignore'):
            df = np.minimum(counts, beta) / beta
        ds = raw_sims * df
        
        self.discount_cache[cache_key] = (raw_sims, counts, df, ds)
        return raw_sims, counts, df, ds

    def pred_discounted(self, u_idx, i_idx, k, beta):
        _, _, _, ds_scores = self.get_discounted_stats(i_idx, beta)
        
        user_ratings = self.matrix[:, u_idx]
        valid_mask = (~np.isnan(user_ratings)) & (ds_scores > 0)
        
        if not np.any(valid_mask): return self.item_means[i_idx]
        
        neighbor_sims = ds_scores[valid_mask]
        neighbor_indices = np.where(valid_mask)[0]
        
        if len(neighbor_sims) > k:
            top_k_args = np.argsort(neighbor_sims)[-k:]
            k_sims = neighbor_sims[top_k_args]
            k_indices = neighbor_indices[top_k_args]
        else:
            k_sims = neighbor_sims
            k_indices = neighbor_indices
            
        neighbor_means = self.item_means[k_indices]
        neighbor_ratings = user_ratings[k_indices]
        deviations = neighbor_ratings - neighbor_means
        
        num = np.sum(k_sims * deviations)
        den = np.sum(np.abs(k_sims))
        
        if den == 0: return self.item_means[i_idx]
        
        pred = self.item_means[i_idx] + (num / den)
        return np.clip(pred, 0.5, 5.0)