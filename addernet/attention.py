import numpy as np

class AdderAttention:
    """
    AdderAttention: A Transformer-like attention mechanism without multiplications.
    Replaces Dot Product with L1 Distance (negative absolute difference).
    Value aggregation uses bitwise gating and additive operations.
    """
    def __init__(self, threshold=None):
        """
        :param threshold: Fixed threshold for bitwise gating of values.
                          If None, dynamic thresholding (mean of scores) is used.
        """
        self.threshold = threshold

    def __call__(self, Q, K, V):
        return self.forward(Q, K, V)

    def forward(self, Q, K, V):
        """
        :param Q: (batch, seq_q, d_model)
        :param K: (batch, seq_k, d_model)
        :param V: (batch, seq_k, d_v)
        :return: (batch, seq_q, d_v)
        """
        # Expand dims for broadcasting
        # Q_exp: (batch, seq_q, 1, d_model)
        # K_exp: (batch, 1, seq_k, d_model)
        Q_exp = np.expand_dims(Q, axis=2)
        K_exp = np.expand_dims(K, axis=1)

        # Calculate relation between Query and Key using L1 distance
        # Score = -|Q - K|
        # Summed over the feature dimension (d_model)
        # score: (batch, seq_q, seq_k)
        score = -np.sum(np.abs(Q_exp - K_exp), axis=-1)

        # Apply attention probabilities to the Value matrix
        # using ONLY additive operations or bitwise gating.
        if self.threshold is not None:
            mask = score >= self.threshold
        else:
            # Dynamic thresholding: keep scores above the mean for each query
            # mean_score: (batch, seq_q, 1)
            mean_score = np.mean(score, axis=-1, keepdims=True)
            mask = score >= mean_score

        # Expand mask to gate Values: (batch, seq_q, seq_k, 1)
        mask_exp = np.expand_dims(mask, axis=-1)
        
        # Expand V to match: (batch, 1, seq_k, d_v)
        V_exp = np.expand_dims(V, axis=1)
        
        # Bitwise gating equivalent for floats (gating to 0 or V)
        # No multiplication used!
        gated_V = np.where(mask_exp, V_exp, 0)
        
        # Additive operation to aggregate
        # output: (batch, seq_q, d_v)
        output = np.sum(gated_V, axis=2)
        
        return output
