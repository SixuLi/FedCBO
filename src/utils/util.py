import torch
import numpy as np

def chunk(a, i, n):
    a2 = chunkify(a, n)
    return a2[i]

def chunkify(a, n):
    # splits list into even size list of lists
    # [1,2,3,4] -> [1,2], [3,4]

    k, m = divmod(len(a), n)
    gen = (a[i * k + min(i, m):(i+1) * k + min(i+1, m)] for i in range(n))
    return list(gen)

