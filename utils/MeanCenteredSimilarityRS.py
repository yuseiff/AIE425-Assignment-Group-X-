import numpy as np

class MeanCentered_Cosine_similarity_RS:
    def __init__(self, matrix):
        # Matrix: Rows=Users, Cols=Items
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
        # 1. Calculate User Means (Row-wise)
        with np.errstate(invalid='ignore'):
            self.means = np.nanmean(self.matrix, axis=1)
            
        # Handle users with no ratings (mean = NaN) -> set to 0
        self.means = np.nan_to_num(self.means)

        # 2. Create Mean-Centered Matrix
        # Reshape means to (N, 1) to broadcast subtraction across columns
        self.matrix_centered = self.matrix - self.means[:, np.newaxis]
        
        # Replace NaNs in centered matrix with 0 for dot product
        # (Missing rating = 0 contribution in centered cosine)
        self.matrix_centered_zeroed = np.nan_to_num(self.matrix_centered)
        
        # 3. Compute Norms of Centered Vectors
        self.norms = np.sqrt(np.sum(self.matrix_centered_zeroed**2, axis=1))
        self.norms[self.norms == 0] = 1e-9 # Avoid division by zero
        
        print(f"Model fitted. Means & Centered Matrix computed for {len(self.means)} users.")

    def compute_similarities(self, u_idx):
        """
        Calculates Mean-Centered Cosine Similarity between User u_idx and ALL other users.
        """
        if u_idx in self.sim_cache:
            return self.sim_cache[u_idx]
            
        if self.matrix_centered is None: raise Exception("Run .fit() first!")
            
        # 1. Get Target Centered Vector
        target_vec = self.matrix_centered_zeroed[u_idx]
        
        # 2. Vectorized Dot Product (Target vs All)
        dot_products = self.matrix_centered_zeroed.dot(target_vec)
        
        # 3. Compute Cosine Sim
        sim_scores = dot_products / (self.norms * self.norms[u_idx])
        
        # 4. Set self-similarity to 0
        sim_scores[u_idx] = 0.0
        
        # Cache and return
        self.sim_cache[u_idx] = sim_scores
        return sim_scores

    def pred(self, u_idx, i_idx, k):
        """
        Prediction using Mean-Centered Formula:
        Pred = Mean_u + (Sum(Sim * (R_vi - Mean_v)) / Sum(|Sim|))
        """
        if self.matrix_centered is None: raise Exception("Run .fit() first!")
        
        # 1. Get Similarities
        sim_scores = self.compute_similarities(u_idx)
        
        # 2. Get Neighbor Ratings (Raw and Centered)
        item_ratings = self.matrix[:, i_idx]
        
        # Valid if: Rated (not NaN) AND Sim > 0 (Positive Correlation)
        valid_mask = (~np.isnan(item_ratings)) & (sim_scores > 0)
        
        if not np.any(valid_mask):
            return self.means[u_idx] # Fallback: User's Average
            
        neighbor_sims = sim_scores[valid_mask]
        neighbor_indices = np.where(valid_mask)[0]
        
        # 3. Top K Neighbors
        if len(neighbor_sims) > k:
            top_k_args = np.argsort(neighbor_sims)[-k:]
            k_sims = neighbor_sims[top_k_args]
            k_indices = neighbor_indices[top_k_args]
        else:
            k_sims = neighbor_sims
            k_indices = neighbor_indices
            
        # 4. Calculate Prediction Term
        # We need (R_vi - Mean_v) which is exactly what matrix_centered stores
        # Note: We use the *centered* value for the neighbor
        k_centered_ratings = self.matrix_centered_zeroed[k_indices, i_idx]
        
        numerator = np.sum(k_sims * k_centered_ratings)
        denominator = np.sum(np.abs(k_sims))
        
        if denominator == 0:
            return self.means[u_idx]
            
        pred = self.means[u_idx] + (numerator / denominator)
        
        # Clip to valid range (0.5 to 5.0)
        return np.clip(pred, 0.5, 5.0)
    

import numpy as np

class Discounted_MeanCentered_Cosine_similarity_RS:
    def __init__(self, matrix):
        # Matrix: Rows=Users, Cols=Items
        self.matrix = np.array(matrix, dtype=float)
        self.means = None
        self.sim_cache = {}
        self.discount_cache = {} # Cache for (raw_sims, counts, df, ds)

    def fit(self):
        """
        Computes User Means (ignoring NaNs).
        """
        # Calculate mean for each row, ignoring NaNs
        with np.errstate(invalid='ignore'): 
            self.means = np.nanmean(self.matrix, axis=1)
            
        # Handle users with no ratings (mean = NaN) -> set to 0
        self.means = np.nan_to_num(self.means)
        print(f"Model fitted. Means computed for {len(self.means)} users.")

    def _calculate_pair_sim_and_count(self, u_idx, v_idx):
        """
        Calculates Pearson Correlation AND Intersection Count.
        Returns: (similarity, count)
        """
        u_vec = self.matrix[u_idx]
        v_vec = self.matrix[v_idx]
        
        # 1. Find Intersection (Common Items)
        mask = ~np.isnan(u_vec) & ~np.isnan(v_vec)
        count = np.sum(mask)
        
        if count == 0:
            return 0.0, 0
            
        # 2. Extract Ratings on Intersection
        u_common = u_vec[mask]
        v_common = v_vec[mask]
        
        # 3. Center the ratings (Subtract User's Overall Mean)
        u_centered = u_common - self.means[u_idx]
        v_centered = v_common - self.means[v_idx]
        
        # 4. Compute Cosine on Centered Vectors
        dot = np.dot(u_centered, v_centered)
        norm_u = np.linalg.norm(u_centered)
        norm_v = np.linalg.norm(v_centered)
        
        if norm_u == 0 or norm_v == 0:
            return 0.0, count
            
        sim = dot / (norm_u * norm_v)
        return sim, count

    def compute_similarities(self, u_idx):
        """
        Calculates Pearson Correlation between User u_idx and ALL other users.
        """
        # Return cached result if available
        if u_idx in self.sim_cache:
            return self.sim_cache[u_idx]
            
        num_users = self.matrix.shape[0]
        sims = np.zeros(num_users)
        
        for v_idx in range(num_users):
            if u_idx == v_idx: continue
            sim, _ = self._calculate_pair_sim_and_count(u_idx, v_idx)
            sims[v_idx] = sim
            
        self.sim_cache[u_idx] = sims
        return sims

    def get_discounted_stats(self, u_idx, beta):
        """
        Calculates DF and DS for Mean-Centered Similarity.
        Returns: (Raw_Sim, Common_Counts, DF, DS)
        """
        if u_idx in self.discount_cache:
            return self.discount_cache[u_idx]

        if self.means is None: raise Exception("Run .fit() first!")

        num_users = self.matrix.shape[0]
        raw_sims = np.zeros(num_users)
        counts = np.zeros(num_users)
        
        # Loop to calculate both Sim and Counts
        for v_idx in range(num_users):
            if u_idx == v_idx: continue
            
            sim, count = self._calculate_pair_sim_and_count(u_idx, v_idx)
            raw_sims[v_idx] = sim
            counts[v_idx] = count
            
        # Calculate DF and DS
        df = np.minimum(counts, beta) / beta
        ds = raw_sims * df
        
        # Cache results
        self.sim_cache[u_idx] = raw_sims
        self.discount_cache[u_idx] = (raw_sims, counts, df, ds)
        
        return raw_sims, counts, df, ds

    def pred(self, u_idx, i_idx, k):
        """Standard Pearson Prediction"""
        return self._predict_generic(u_idx, i_idx, k, use_discounted=False)
        
    def pred_discounted(self, u_idx, i_idx, k, beta):
        """Discounted Pearson Prediction"""
        # Ensure stats are computed/cached
        self.get_discounted_stats(u_idx, beta)
        return self._predict_generic(u_idx, i_idx, k, use_discounted=True)

    def _predict_generic(self, u_idx, i_idx, k, use_discounted=False):
        if self.means is None: raise Exception("Run .fit() first!")
        
        # 1. Get Similarities (Raw or Discounted)
        if use_discounted:
            # We assume get_discounted_stats has been called or we look up cache
            # For safety, pull from cache (logic requires it to be populated)
             if u_idx in self.discount_cache:
                 sim_scores = self.discount_cache[u_idx][3] # Index 3 is DS
             else:
                 # Should not happen if called via pred_discounted, but fallback
                 return self.means[u_idx]
        else:
            sim_scores = self.compute_similarities(u_idx)
        
        # 2. Get Neighbor Ratings
        item_ratings = self.matrix[:, i_idx]
        valid_mask = (~np.isnan(item_ratings)) & (sim_scores > 0)
        
        if not np.any(valid_mask): return self.means[u_idx]
            
        neighbor_sims = sim_scores[valid_mask]
        neighbor_indices = np.where(valid_mask)[0]
        
        # 3. Top K
        if len(neighbor_sims) > k:
            top_k_args = np.argsort(neighbor_sims)[-k:]
            k_sims = neighbor_sims[top_k_args]
            k_indices = neighbor_indices[top_k_args]
        else:
            k_sims = neighbor_sims
            k_indices = neighbor_indices
            
        # 4. Formula: Mean_u + Sum(Sim * (R_vi - Mean_v)) / Sum(|Sim|)
        neighbor_means = self.means[k_indices]
        neighbor_ratings = item_ratings[k_indices]
        centered_ratings = neighbor_ratings - neighbor_means
        
        numerator = np.sum(k_sims * centered_ratings)
        denominator = np.sum(np.abs(k_sims))
        
        if denominator == 0: return self.means[u_idx]
            
        pred = self.means[u_idx] + (numerator / denominator)
        return np.clip(pred, 0.5, 5.0)