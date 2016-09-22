import numpy as np

from collections import MutableMapping

class Bidict(MutableMapping):
    """Bidirectional dictionary.

    Implementation following in part https://github.com/jab/bidict/
    """

    def __init__(self, *args, **kwargs):
        self._fwd = dict()
        self._inv = dict()
        self.update(dict(*args, **kwargs))
        self.inv = object.__new__(Bidict)
        self.inv._fwd = self._inv
        self.inv._inv = self._fwd
        self.inv.inv = self
        
    def __getitem__(self, key):
        key = _hashable(key)
        if key in self._fwd:
            return self._fwd[key]
        else:
            return self.__missing__(key)
        
    def __setitem__(self, key, value):
        if key in self._fwd:
            raise ValueError('key {} exists'.format(key))
        if _hashable(value) in self._inv:
            raise ValueError('value {} exists'.format(value))
        self._fwd[key] = value
        self._inv[_hashable(value)] = key
        
    def __delitem__(self, key):
        value = self._fwd[key]
        del self._fwd[key]
        del self._inv[value]

    def __contains__(self, key):
        return key in self._fwd
        
    def __iter__(self):
        return iter(self._fwd)

    def __len__(self):
        return len(self._fwd)

    def __missing__(self, key):
        raise KeyError(key)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, repr(self._fwd))

class DefaultBidict(Bidict):
    """Bidirectional default dictionary."""

    def __init__(self, default_factory=None, *args, **kwargs):
        super(DefaultBidict, self).__init__(*args, **kwargs)
        self.default_factory = default_factory

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        value = self.default_factory()
        self[key] = value
        return value
        
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.default_factory, repr(self._fwd))
        
class IncBidict(DefaultBidict):
    """Bidirectional dictionary defaulting to max value plus one."""

    def __init__(self, *args, **kwargs):
        self._max = -1
        super(IncBidict, self).__init__(lambda: self._max+1, *args, **kwargs)

    def __setitem__(self, key, value):
        super(IncBidict, self).__setitem__(key, value)
        self._max = max(self._max, value)

def _hashable(key):
    """Return key or hashable equivalent if unhashable."""
    # This was introduced as a workaround to allow Bidicts to contain
    # numpy ndarray values. ndarray is not hashable but can be
    # represented by a tuple for this purpose.
    if not isinstance(key, np.ndarray):
        return key
    else:
        return tuple(_hashable(k) for k in key)
