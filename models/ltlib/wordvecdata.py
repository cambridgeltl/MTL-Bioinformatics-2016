import numpy as np

from collections import OrderedDict
from os import path

def load(filename, max_rank=None, vocabulary=None):
    """Load word vector data.

    Args:
        filename: name of file to load.
        max_rank: maximum number of embeddings to load (None for no limit)
        vocabulary: words to load embeddings for (None for all)

    Returns:
        Tuple (word_to_idx, embedding) where word_to_idx is
        an OrderedDict mapping words to integer indices and embedding is
        a numpy array of shape (word-count, vector-size).
    """
    # TODO support for other formats
    return load_w2v_binary(filename, max_rank, vocabulary)

def load_w2v_binary(filename, max_rank=None, vocabulary=None):
    """Load word2vec binary format.

    Args:
        filename: name of file to load.
        max_rank: maximum number of embeddings to load (None for no limit)
        vocabulary: words to load embeddings for (None for all)

    Returns:
        Tuple (word_to_idx, embedding) where word_to_idx is an
        OrderedDict mapping words to integer indices and embedding is
        a numpy array of shape (word-count, vector-size).
    """
    word_to_idx, vectors = OrderedDict(), []
    with open(filename, 'rb') as f:
        # header has vocab and vector sizes as strings
        word_count, vec_size = map(int, f.readline().split())
        for i in range(word_count):
            if max_rank and i > max_rank:
                break
            word_to_idx[read_w2v_word(f)] = len(word_to_idx)
            vectors.append(np.fromfile(f, np.float32, vec_size))
    vectors = np.array(vectors)
    if vocabulary is not None:
        word_to_idx, vectors = filter_words(word_to_idx, vectors, vocabulary)
    return word_to_idx, vectors

def read_w2v_word(flo):
    """Return word from file-like object, break on space."""
    # http://docs.python.org/2/library/functions.html#iter
    word = ''.join(iter(lambda: flo.read(1), ' '))
    return word.lstrip('\n')     # harmonize w2v format variants

def filter_words(word_to_idx, vectors, vocabulary):
    """Filter word vector data to vocabulary."""
    filtered_to_idx, filtered_indices = OrderedDict(), []
    for word, idx in word_to_idx.items():
        if word in vocabulary:
            filtered_to_idx[word] = len(filtered_to_idx)
            filtered_indices.append(idx)
    return filtered_to_idx, vectors[filtered_indices]
