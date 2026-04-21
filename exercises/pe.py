import numpy as np
import pdb
vocab_size = 1000
d_model = 512

embeddings = np.random.randn(vocab_size, d_model)
token_ids = np.random.randint(0, vocab_size, size=10)
x = embeddings[token_ids]  # shape: (10, 512)

def sinusoidal_pe(emb, d_model):
    # pdb.set_trace()   
    seq_len = len(emb)
    pe = np.zeros_like(emb)
    pos = np.arange(seq_len)[:, np.newaxis]
    div = np.exp(np.arange(0, d_model, 2) * -np.log(10000.0) / d_model)

    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    
    return emb + pe

def rope(emb, d_model):
    seq_len = len(emb)
    
    theta = np.exp(np.arange(0, d_model, 2) * -np.log(10000.0) / d_model)
    
    positions = np.arange(seq_len)[:, np.newaxis]
    angles = positions * theta
    
    x0 = emb[:, 0::2]  # even dimensions
    x1 = emb[:, 1::2]  # odd dimensions
    
    rotated_even = x0 * np.cos(angles) - x1 * np.sin(angles)
    rotated_odd  = x0 * np.sin(angles) + x1 * np.cos(angles)
    
    result = np.zeros_like(emb)
    result[:, 0::2] = rotated_even
    result[:, 1::2] = rotated_odd
    
    return result

    

result = rope(x, d_model)
print(result.shape)



 