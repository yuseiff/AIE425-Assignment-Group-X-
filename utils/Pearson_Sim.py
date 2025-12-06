import numpy as np

class Pearson_Correlation_RS:
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=float)
        self.global_means = None 
        self.sim_cache = {}

    def fit(self):
        with np.errstate(invalid='ignore'):
            self.global_means = np.nanmean(self.matrix, axis=1)
        self.global_means = np.nan_to_num(self.global_means)
        print(f"Pearson Model fitted. Global means computed for {len(self.global_means)} users.")

    def _calculate_pair_pearson(self, u_idx, v_idx):
        u_vec = self.matrix[u_idx]
        v_vec = self.matrix[v_idx]
        mask = ~np.isnan(u_vec) & ~np.isnan(v_vec)
        if np.sum(mask) < 2: return 0.0
        u_common = u_vec[mask]
        v_common = v_vec[mask]
        u_centered = u_common - np.mean(u_common)
        v_centered = v_common - np.mean(v_common)
        dot = np.dot(u_centered, v_centered)
        norm_u = np.linalg.norm(u_centered)
        norm_v = np.linalg.norm(v_centered)
        if norm_u == 0 or norm_v == 0: return 0.0
        return dot / (norm_u * norm_v)

    def compute_similarities(self, u_idx):
        if u_idx in self.sim_cache: return self.sim_cache[u_idx]
        num_users = self.matrix.shape[0]
        sims = np.zeros(num_users)
        for v_idx in range(num_users):
            if u_idx == v_idx: continue
            sims[v_idx] = self._calculate_pair_pearson(u_idx, v_idx)
        self.sim_cache[u_idx] = sims
        return sims

    def pred(self, u_idx, i_idx, k):
        if self.global_means is None: raise Exception("Run .fit() first!")
        sim_scores = self.compute_similarities(u_idx)
        item_ratings = self.matrix[:, i_idx]
        valid_mask = (~np.isnan(item_ratings)) & (sim_scores > 0)
        if not np.any(valid_mask): return self.global_means[u_idx]
        neighbor_sims = sim_scores[valid_mask]
        neighbor_indices = np.where(valid_mask)[0]
        if len(neighbor_sims) > k:
            top_k_args = np.argsort(neighbor_sims)[-k:]
            k_sims = neighbor_sims[top_k_args]
            k_indices = neighbor_indices[top_k_args]
        else:
            k_sims = neighbor_sims
            k_indices = neighbor_indices
        neighbor_globals = self.global_means[k_indices]
        neighbor_ratings = item_ratings[k_indices]
        deviations = neighbor_ratings - neighbor_globals
        numerator = np.sum(k_sims * deviations)
        denominator = np.sum(np.abs(k_sims))
        if denominator == 0: return self.global_means[u_idx]
        pred = self.global_means[u_idx] + (numerator / denominator)
        return np.clip(pred, 0.5, 5.0)

class Discounted_Pearson_RS(Pearson_Correlation_RS):
    def __init__(self, matrix):
        # Override init to add discount_cache
        super().__init__(matrix)
        self.discount_cache = {}

    def _calculate_pair_pearson_and_count(self, u_idx, v_idx):
        u_vec = self.matrix[u_idx]
        v_vec = self.matrix[v_idx]
        mask = ~np.isnan(u_vec) & ~np.isnan(v_vec)
        count = np.sum(mask)
        if count < 2: return 0.0, count
        u_common = u_vec[mask]
        v_common = v_vec[mask]
        u_centered = u_common - np.mean(u_common)
        v_centered = v_common - np.mean(v_common)
        dot = np.dot(u_centered, v_centered)
        norm_u = np.linalg.norm(u_centered)
        norm_v = np.linalg.norm(v_centered)
        if norm_u == 0 or norm_v == 0: return 0.0, count
        return (dot / (norm_u * norm_v)), count

    def get_discounted_stats(self, u_idx, beta):
        # --- CACHE CHECK (The Fix) ---
        cache_key = (u_idx, beta)
        if cache_key in self.discount_cache:
            return self.discount_cache[cache_key]

        num_users = self.matrix.shape[0]
        raw_sims = np.zeros(num_users)
        counts = np.zeros(num_users)
        
        for v_idx in range(num_users):
            if u_idx == v_idx: continue
            sim, count = self._calculate_pair_pearson_and_count(u_idx, v_idx)
            raw_sims[v_idx] = sim
            counts[v_idx] = count
            
        with np.errstate(divide='ignore', invalid='ignore'):
            df = np.minimum(counts, beta) / beta
        ds = raw_sims * df
        
        # Save to Cache
        self.discount_cache[cache_key] = (raw_sims, counts, df, ds)
        return raw_sims, counts, df, ds

    def pred_discounted(self, u_idx, i_idx, k, beta):
        if self.global_means is None: raise Exception("Run .fit() first!")
        # This will now hit the cache after the first item!
        _, _, _, ds_scores = self.get_discounted_stats(u_idx, beta)
        
        item_ratings = self.matrix[:, i_idx]
        valid_mask = (~np.isnan(item_ratings)) & (ds_scores > 0)
        if not np.any(valid_mask): return self.global_means[u_idx]
        neighbor_sims = ds_scores[valid_mask]
        neighbor_indices = np.where(valid_mask)[0]
        if len(neighbor_sims) > k:
            top_k_args = np.argsort(neighbor_sims)[-k:]
            k_sims = neighbor_sims[top_k_args]
            k_indices = neighbor_indices[top_k_args]
        else:
            k_sims = neighbor_sims
            k_indices = neighbor_indices
        neighbor_globals = self.global_means[k_indices]
        neighbor_ratings = item_ratings[k_indices]
        deviations = neighbor_ratings - neighbor_globals
        numerator = np.sum(k_sims * deviations)
        denominator = np.sum(np.abs(k_sims))
        if denominator == 0: return self.global_means[u_idx]
        pred = self.global_means[u_idx] + (numerator / denominator)
        return np.clip(pred, 0.5, 5.0)