import re
import numpy as np

import wordvecdata 

from logging import warn
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from bidict import IncBidict

def uniform(shape, scale=0.05):
    # TODO move to more sensible place, avoid redundancy with
    # keras.initializations.uniform
    return np.random.uniform(low=-scale, high=scale, size=shape)

class FeatureMapping(object):
    """Abstract base class for feature mappings."""

    __metaclass__ = ABCMeta
    
    def __init__(self, name):
        """Initialize FeatureMapping.

        Args:
            name: feature name. Used as default key to feature dict
                in DataItem.
        """
        self.name = name
    
    @abstractmethod
    def __call__(self, dataitem):
        """Return value of feature for given DataItem."""
        pass

    @abstractmethod
    def invert(self, value):
        """Return string representation of feature value."""
        pass

class IndexedFeatureMapping(FeatureMapping):
    """Feature mapping from discrete keys to integer indices."""

    # Data key to use by default
    default_data_key = 'text'

    def __init__(self, index_map=None, data_key=None, name=None):
        """Initialize IndexedFeatureMapping.

        Args:
            index_map: mapping from values to indices.
            data_key: key identifying the DataItem data to map. If None,
                defaults to default_data_key.
        """
        super(IndexedFeatureMapping, self).__init__(name)
        if index_map is not None:
            self.index_map = IncBidict(index_map)
        else:
            self.index_map = IncBidict()
        if data_key is None:
            data_key = self.default_data_key
        self.data_key = data_key
        
    def __call__(self, dataitem):
        """Return value of feature for given DataItem."""
        return self.index_map[dataitem.data[self.data_key]]

    def __getitem__(self, key):
        """Direct mapping lookup."""
        return self.index_map[key]

    def __len__(self):
        return len(self.index_map)

    def invert(self, index):
        return self.index_map.inv[index]

class EmbeddingFeature(IndexedFeatureMapping):
    """Feature mapping to indices with associated vector values."""

    # Missing key value to use by default.
    default_missing_key = None

    # Required vocabulary items (other than the missing key value).
    required_vocabulary = []

    def __init__(self, index_map=None, weights=None, output_dim=None,
                 init=uniform, missing_key=None, data_key=None, name=None):
        """Initialize EmbeddingFeature.

        Either initial weights or output dimension must be provided.

        Args:
            index_map: mapping from values to indices.
            weights: array-like of initial embedding weights.
            output_dim: embedding dimension.
            init: initialization function to use for embeddings. Only
                used when not specified in weights.
            missing_key: key to use for lookup for keys not in index_map.
                If None, extend index_map as needed.
            data_key: key identifying the DataItem data to map. If None,
                defaults to default_data_key.
        """
        super(EmbeddingFeature, self).__init__(
            index_map=index_map, data_key=data_key, name=name,
        )
        if weights is None:
            self._weights = None
            self._output_dim = output_dim
        else:
            self._weights = np.asarray(weights)
            self._output_dim = self._weights.shape[1]
        if self.output_dim is None:
            raise ValueError('could not determine embedding dimension')
        if output_dim is not None and output_dim != self.output_dim:
            raise ValueError('inconsistent embedding dimension')
        if missing_key is None:
            missing_key = self.default_missing_key
        self._init = init
        self._missing_key = missing_key
        self.total_count = 0
        self.missing_count = 0
        self.oov_count = defaultdict(int)

    def __call__(self, dataitem):
        key = dataitem.data[self.data_key]
        key = self.normalize(key, self.index_map)
        if key not in self.index_map:
            key = self.__missing__(key)
        self.total_count += 1
        return self.index_map[key]

    def __missing__(self, key):
        # TODO reconsider special function for this.
        self.missing_count += 1
        self.oov_count[key] += 1
        if self._missing_key is not None:
            key = self._missing_key
        return key

    @property
    def weights(self):
        if self._weights is None:
            return self._init((self.input_dim, self.output_dim))
        elif self.input_dim <= self._weights.shape[0]:
            return self._weights
        else:
            # Partial weights, add in newly initialized for missing.
            missing = self.input_dim - self._weights.shape[0]
            warn('incomplete weights, added {} missing'.format(missing))
            return np.concatenate([self._weights,
                                   self._init((missing, self.output_dim))])

    @property
    def input_dim(self):
        return len(self.index_map)

    @property
    def output_dim(self):
        return self._output_dim

    def average_weight(self):
        """Return the average weight vector length."""
        if self._weights is None:
            warn('average_weight: no weights')
            return 0.0
        else:
            return np.average([np.linalg.norm(w) for w in self._weights])

    def missing_rate(self):
        """Return the ratio of missing to total lookups."""
        return 1.*self.missing_count/self.total_count

    def most_frequent_oov(self, max_rank=5):
        freq_oov = [(v, k) for k, v in self.oov_count.items()]
        return sorted(freq_oov, reverse=True)[:max_rank]

    def summary(self):
        """Return string summarizing embedding statistics."""
        return ('Vocab {} words, avg wv len {}, OOV {:.2%} ({}/{}) '
                '(top OOV: {})'.format(
            len(self), self.average_weight(), self.missing_rate(),
            self.missing_count, self.total_count,
            ' '.join('{} ({})'.format(w, n)
                     for n, w in self.most_frequent_oov())
        ))

    @classmethod
    def from_file(cls, filename, max_rank=None, vocabulary=None, name=None,
                  add_missing=False, **kwargs):
        index_map, weights = wordvecdata.load(filename, max_rank)
        if vocabulary is not None:
            # Filter to vocabulary, taking normalization into account.
            vocabulary = set(cls.normalize(w, index_map) for w in vocabulary)
            # Make sure the missing key value is included in the vocabulary.
            missing_key = kwargs.get('missing_key', cls.default_missing_key)
            if missing_key is not None:
                vocabulary.add(missing_key)
            # ... and other required vocab items (TODO: clean up logic)
            for w in cls.required_vocabulary:
                vocabulary.add(w)
            index_map, weights = wordvecdata.filter_words(index_map, weights,
                                                          vocabulary)
        obj = cls(index_map=index_map, weights=weights, name=name, **kwargs)
        if vocabulary is not None and add_missing:
            # add missing vocabulary items to embedding
            for v in vocabulary:
                obj[v]
        return obj

    @staticmethod
    def normalize(key, vocabulary):
        """Return form of key to use for lookup in vocabulary."""
        # Static method to allow normalization to apply to vocabulary
        # filtering in from_file() before initialization
        return key

class NormEmbeddingFeature(EmbeddingFeature):
    """Embedding lookup feature with normalization."""

    default_missing_key = 'UNKNOWN'
    required_vocabulary = ['PADDING']

    @staticmethod
    def normalize(key, vocabulary):
        orig_key = key
        # Normalize as fallback if direct lookup fails
        for norm in (lambda s: s.lower(),
                     lambda s: re.sub(r'[+-]?(?:[.,]?[0-9])+', '0', s)):
            if key in vocabulary:
                return key
            key = norm(key)
        # Return original for missing for accurate OOV stats
        return orig_key

class SennaEmbeddingFeature(EmbeddingFeature):
    """Embedding lookup feature with SENNA-like normalization."""

    default_missing_key = 'UNKNOWN'
    required_vocabulary = ['PADDING']

    @staticmethod
    def normalize(key, vocabulary):
        # No change for direct hits
        if key in vocabulary:
            return key
        # SENNA normalization: lowercase and replace numbers with "0"
        return re.sub(r'[+-]?(?:[.,]?[0-9])+', '0', key.lower())

class SennaCapsFeature(EmbeddingFeature):
    """Token capitalization feature using SENNA categories."""

    def __init__(self, data_key='text', name=None, output_dim=5):
        super(SennaCapsFeature, self).__init__(
            output_dim=output_dim, data_key=data_key, name=name
        )

    def __call__(self, dataitem):
        text = dataitem.data[self.data_key]
        if text.isupper():
            category = 'allcaps'
        elif text[0].isupper():
            category = 'initcap'
        elif any(c.isupper() for c in text):
            category = 'hascap'
        else:
            category = 'nocaps'
        return self[category]

# TODO: not sure this belongs here. Unlike the feature mappings above,
# this assumes quite a lot of knowledge of DataItem structure.
class WindowedInput(object):
    """Catenate feature values in a window of items in a sequence."""

    def __init__(self, window_size, padding, key):
        if window_size % 2 == 0:
            raise ValueError('window size must be odd')
        self.window_size = window_size
        self.padding = padding
        self.key = key

    def __call__(self, dataitem):
        windowed = []
        half_win = (self.window_size - 1) / 2
        for offset in range(-half_win, half_win+1):
            sibling = dataitem.sibling(offset)
            if sibling is not None:
                windowed.append(sibling.feature[self.key])
            else:
                windowed.append(self.padding)
        return np.array(windowed)

    @property
    def input_length(self):
        return self.window_size

    @property
    def shape(self):
        """Return shape of generated input."""
        # TODO this depends on the base features, don't assume fixed size
        return (self.input_length,)

def windowed_inputs(window_size, features, padding_key='PADDING'):
    """Return list of WindowedInputs, one for each given Feature.

    Given Features must support indexing by padding_key.
    """
    return [
        WindowedInput(window_size, f[padding_key], f.name) for f in features
    ]
