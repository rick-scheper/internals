import numpy as np, time

T, D = 512, 64      # sequence length, head_dim
np.random.seed(0)

K_cache = np.random.randn(T,D).astype(np.float32)
V_cache = np.random.randn(T,D).astype(np.float32)

def attention_no_cache(Q_all, K_all, V_all):
    scores = Q_all @ K_all.T / np.sqrt(D)
    weights = np.exp(scores) / np.exp(scores).sum(-1, keepdims=True)
    return weights @ V_all

def attention_with_cache(q_new, K_cache, V_cache):
    scores = q_new @K_cache.T / np.sqrt(D)
    weights = np.exp(scores) / np.exp(scores).sum()
    return weights @ V_cache

Q_all = np.random.randn(T,D).astype(np.float32)
q_new = Q_all[-1:,:]

t0 = time.perf_counter()
for _ in range(1000): attention_no_cache(Q_all, K_cache, V_cache)
print(f'No cache: {(time.perf_counter()-t0)*1000:.1f}ms per 1k calls')
t0 = time.perf_counter()
for _ in range(1000): attention_with_cache(q_new, K_cache, V_cache)
print(f'With cache: {(time.perf_counter()-t0)*1000:.1f}ms per 1k calls')

