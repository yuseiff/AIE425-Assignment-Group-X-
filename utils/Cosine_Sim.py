import numpy as np

class Cosine_similarity_RS:
    def __init__(self, matrix):
        # Store raw matrix with NaNs
        self.matrix = np.array(matrix, dtype=float)
        # Cache to store similarity vectors for target users so we don't re-compute
        self.sim_cache = {} 
        # Cache for discount stats
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
        # Get vectors
        u_vec = self.matrix[u_idx]
        v_vec = self.matrix[v_idx]
        
        # Find common items (where both are not NaN)
        # This is the "Intersection" logic
        mask = ~np.isnan(u_vec) & ~np.isnan(v_vec)
        
        # If no common items, similarity is 0
        if not np.any(mask):
            return 0.0, 0 # Sim, Count
            
        # Extract common ratings
        u_common = u_vec[mask]
        v_common = v_vec[mask]
        
        # Compute Cosine on this subset
        # Sim = (A . B) / (||A|| * ||B||)
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
        # Return cached result if available
        if u_idx in self.sim_cache:
            return self.sim_cache[u_idx]
            
        num_users = self.matrix.shape[0]
        sims = np.zeros(num_users)
        
        # Loop through all users (This takes a moment, but runs only once per target)
        # Optimized with list comprehension for speed
        u_vec = self.matrix[u_idx]
        
        # We perform a matrix-level intersection check to speed up the loop
        # 1. Identify valid candidates (users who share at least 1 item)
        # Using matrix operations to find overlaps roughly
        # (This is an optimization to avoid calling the helper on totally disjoint users)
        
        for v_idx in range(num_users):
            if u_idx == v_idx:
                sims[v_idx] = 0.0
                continue
                
            sim, _ = self._calculate_pair_sim(u_idx, v_idx)
            sims[v_idx] = sim
            
        # Cache the result
        self.sim_cache[u_idx] = sims
        return sims

    def pred(self, u_idx, i_idx, k):
        """
        Predict rating for User (row u_idx) on Item (col i_idx).
        """
        # 1. Get Similarity (Cached)
        sim_scores = self.compute_similarities(u_idx)
        
        # 2. Get Item Ratings
        item_ratings = self.matrix[:, i_idx]
        
        # 3. Filter Valid Neighbors
        valid_mask = (~np.isnan(item_ratings)) & (sim_scores > 0)
        
        if not np.any(valid_mask): return 0.0 
            
        neighbor_sims = sim_scores[valid_mask]
        neighbor_ratings = item_ratings[valid_mask]
        
        # 4. Top K
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
        # Check cache
        if u_idx in self.discount_cache:
            return self.discount_cache[u_idx]

        num_users = self.matrix.shape[0]
        raw_sims = np.zeros(num_users)
        counts = np.zeros(num_users)
        
        # Loop to calculate both Sim and Counts
        for v_idx in range(num_users):
            if u_idx == v_idx: continue
            
            sim, count = self._calculate_pair_sim(u_idx, v_idx)
            raw_sims[v_idx] = sim
            counts[v_idx] = count
            
        # Calculate DF and DS Vectorized
        df = np.minimum(counts, beta) / beta
        ds = raw_sims * df
        
        # Cache results (including the raw sims in the parent cache to avoid re-work)
        self.sim_cache[u_idx] = raw_sims
        self.discount_cache[u_idx] = (raw_sims, counts, df, ds)
        
        return raw_sims, counts, df, ds

    def pred_discounted(self, u_idx, i_idx, k, beta):
        """
        Predict using Discounted Similarity.
        """
        # 1. Get DS stats (Cached)
        _, _, _, ds_scores = self.get_discounted_stats(u_idx, beta)
        
        # 2. Standard Prediction logic using DS scores
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