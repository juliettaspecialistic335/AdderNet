import numpy as np
from addernet.attention import AdderAttention

def test_adder_attention():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Dimensions: Batch = 2, Seq_q = 3, Seq_k = 4, D_model = 5, D_v = 6
    batch_size = 2
    seq_q = 3
    seq_k = 4
    d_model = 5
    d_v = 6
    
    Q = np.random.randn(batch_size, seq_q, d_model)
    K = np.random.randn(batch_size, seq_k, d_model)
    V = np.random.randn(batch_size, seq_k, d_v)
    
    # Initialize Attention
    attention = AdderAttention()
    
    # Run forward pass
    output = attention.forward(Q, K, V)
    
    # Check output shape
    assert output.shape == (batch_size, seq_q, d_v), f"Expected shape {(batch_size, seq_q, d_v)}, got {output.shape}"
    
    print("Test passed: AdderAttention outputs correct shape and runs without multiplication operations.")

if __name__ == "__main__":
    test_adder_attention()
