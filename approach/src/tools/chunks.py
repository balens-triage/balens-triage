import math


def chunks_n_sized(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def n_chunks(l, n):
    chunk_size = math.ceil(len(l) / n)
    return chunks_n_sized(l, chunk_size)
