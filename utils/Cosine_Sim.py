import numpy as np

class Cosine_similarity_RS:
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=float)
        self.sim_cache = {} 
        self.discount_cache = {}

    def fit(self):
        """
        No pre-computation needed for Intersection-Based Lazy approach.
        """
        print(f"Model initialized for {self.matrix.shape[0]} users. Ready for lazy similarity calculation.")

    def _calculate_pair_sim(self, u_idx, v_idx):
        """
        Internal Helper: Calculates Standard Cosine Sim between two users 
        based ONLY on their COMMON (intersection) items.
        """
        u_vec = self.matrix[u_idx]
        v_vec = self.matrix[v_idx]
        
        mask = ~np.isnan(u_vec) & ~np.isnan(v_vec)
        
        if not np.any(mask):
            return 0.0, 0 
            
        u_common = u_vec[mask]
        v_common = v_vec[mask]
        
        dot = np.dot(u_common, v_common)
        norm_u = np.linalg.norm(u_common)
        norm_v = np.linalg.norm(v_common)
        
        if norm_u == 0 or norm_v == 0:
            return 0.0, np.sum(mask)
            
        sim = dot / (norm_u * norm_v)
        return sim, np.sum(mask)

    def compute_similarities(self, u_idx):
        """
        Calculates similarity between User u_idx and ALL other users.
        Uses Caching to ensure performance.
        """
        if u_idx in self.sim_cache:
            return self.sim_cache[u_idx]
            
        num_users = self.matrix.shape[0]
        sims = np.zeros(num_users)
        
    
        u_vec = self.matrix[u_idx]
        
        for v_idx in range(num_users):
            if u_idx == v_idx:
                sims[v_idx] = 0.0
                continue
                
            sim, _ = self._calculate_pair_sim(u_idx, v_idx)
            sims[v_idx] = sim
            
        self.sim_cache[u_idx] = sims
        return sims

    def pred(self, u_idx, i_idx, k):
        """
        Predict rating for User (row u_idx) on Item (col i_idx).
        """
        sim_scores = self.compute_similarities(u_idx)
        
        item_ratings = self.matrix[:, i_idx]
        
        valid_mask = (~np.isnan(item_ratings)) & (sim_scores > 0)
        
        if not np.any(valid_mask): return 0.0 
            
        neighbor_sims = sim_scores[valid_mask]
        neighbor_ratings = item_ratings[valid_mask]
        
        if len(neighbor_sims) > k:
            top_k_idx = np.argsort(neighbor_sims)[-k:]
            k_sims = neighbor_sims[top_k_idx]
            k_ratings = neighbor_ratings[top_k_idx]
        else:
            k_sims = neighbor_sims
            k_ratings = neighbor_ratings
            
        if np.sum(k_sims) == 0: return 0.0
        return np.sum(k_sims * k_ratings) / np.sum(k_sims)


class Discounted_Cosine_similarity_RS(Cosine_similarity_RS):
    """
    Inherits from the standard class but adds Discounted Similarity logic.
    """
    def get_discounted_stats(self, u_idx, beta):
        """
        Returns: (Raw_Sim, Common_Counts, DF, DS)
        """
        if u_idx in self.discount_cache:
            return self.discount_cache[u_idx]

        num_users = self.matrix.shape[0]
        raw_sims = np.zeros(num_users)
        counts = np.zeros(num_users)
        
        for v_idx in range(num_users):
            if u_idx == v_idx: continue
            
            sim, count = self._calculate_pair_sim(u_idx, v_idx)
            raw_sims[v_idx] = sim
            counts[v_idx] = count
            
        df = np.minimum(counts, beta) / beta
        ds = raw_sims * df
        
        self.sim_cache[u_idx] = raw_sims
        self.discount_cache[u_idx] = (raw_sims, counts, df, ds)
        
        return raw_sims, counts, df, ds

    def pred_discounted(self, u_idx, i_idx, k, beta):
        """
        Predict using Discounted Similarity.
        """
        _, _, _, ds_scores = self.get_discounted_stats(u_idx, beta)
        
        item_ratings = self.matrix[:, i_idx]
        valid_mask = (~np.isnan(item_ratings)) & (ds_scores > 0)
        
        if not np.any(valid_mask): return 0.0 
        
        neighbor_sims = ds_scores[valid_mask]
        neighbor_ratings = item_ratings[valid_mask]
        
        if len(neighbor_sims) > k:
            top_k_idx = np.argsort(neighbor_sims)[-k:]
            k_sims = neighbor_sims[top_k_idx]
            k_ratings = neighbor_ratings[top_k_idx]
        else:
            k_sims = neighbor_sims
            k_ratings = neighbor_ratings
            
        if np.sum(k_sims) == 0: return 0.0
        return np.sum(k_sims * k_ratings) / np.sum(k_sims)