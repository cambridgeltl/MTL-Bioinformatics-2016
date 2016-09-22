import numpy as np

from itertools import chain
from util import unique, binarize_sparse, as_scalar

from bidict import Bidict

# TODO: it's strange (but convenient) that each DataItem holds target_map.
# At least for TreeDataItems, this could alternatively be accessed via the
# root. Reconsider the design.

class DataItem(object):
    """Base class for data items.

    Attributes:
        data: dict mapping to data content of DataItem (e.g. token text).
        target_str: target string for DataItem (e.g. POS tag).
        target: numeric representation of target.
        prediction_str: predicted string for DataItem.
        prediction: numeric representation of prediction.
        feature: dict mapping feature types to values.
        input: dict mapping feature types to values.
        target_map: mapping from target string to numeric representation.
    """

    def __init__(self, data=None, target_str=None):
        self._data = data
        self._target_str = target_str
        self.target = None
        self.prediction_str = None
        self.prediction = None
        self.feature = {}
        self.input = {}
        self._target_map = {}

    @property
    def data(self):
        return self._data

    @property
    def target_str(self):
        return self._target_str

    @property
    def target_map(self):
        # TODO consider making map immutable.
        return self._target_map

    def set_target(self, target):
        self.target = target

    def set_prediction(self, prediction):
        self.prediction = prediction

    def set_prediction_str(self, prediction_str):
        self.prediction_str = prediction_str

    def add_feature(self, function, key):
        self.feature[key] = function(self)

    def add_input(self, function, key):
        self.input[key] = function(self)

    def set_target_map(self, target_map, map_target=True):
        """Set map from target string to numeric representation.

        Args:
            map_target: if True, set target based on map.
        """
        if map_target:
            self.set_target(target_map[self.target_str])
        self._target_map = target_map

class TreeDataItem(DataItem):
    """Base class for DataItems that have a tree structure.

    Provides an immutable sequence (abc.Sequence) interface over
    children.

    Attributes:
        parent: the parent TreeDataItem.
        position: the position of the TreeDataItem in parent.
        children: child TreeDataItems.
    """

    def __init__(self, data=None, target_str=None, parent=None, position=None,
                 children=None):
        super(TreeDataItem, self).__init__(data=data, target_str=target_str)
        if children is None:
            children = []
        self.parent = parent
        self.position = position
        self.children = []
        self.add_children(children)

    def add_child(self, child):
        child.parent = self
        child.position = len(self.children)
        self.children.append(child)

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    def sibling(self, offset):
        """Return sibling at given offset or None if one does not exist."""
        siblings = self.parent.children
        position = self.position + offset
        if position < 0 or position >= len(siblings):
            return None
        else:
            return siblings[position]

    def root(self):
        """Return root of TreeDataItem tree."""
        if self.parent is None:
            return self
        else:
            return self.parent.root()

    def find(self, instance_of=None):
        """Return DataItemSequence of selected nodes in tree.

        Args:
            instance_of: if not Null, only return nodes that are instances
                of the given class.
        """
        constraints = []
        if instance_of is not None:
            constraints.append(lambda n: isinstance(n, instance_of))
        return DataItemSequence(traverse(self, yield_if=constraints))

    def __getitem__(self, key):
        """Return child at given index."""
        return self.children[key]

    def __len__(self):
        """Return number of children."""
        return len(self.children)

    def __iter__(self):
        """Iterate over children."""
        return iter(self.children)

def traverse(node, yield_if=None, pre_order=False):
    """Iterate over TreeDataItem nodes.

    Args:
        yield_if: list of functions. Only yield nodes that return True
            for all functions.
        pre_order: if True, yield node before its children (pre-order
            traversal), otherwise, yield children first (post-order).
    """
    if yield_if is None:
        yield_node = lambda _: True
    else:
        yield_node = lambda n: all(f(n) for f in yield_if)
    if pre_order:
        if yield_node(node):
            yield node
    for child in node.children:
        for n in traverse(child, yield_if=yield_if, pre_order=pre_order):
            yield n
    if not pre_order:
        if yield_node(node):
            yield node

class DataItemSequence(object):
    """An immutable sequence of DataItems.

    Makes attributes of contained DataItems available as sequences of
    appropriate types (numpy array for numeric, list for others).
    """

    # A key point of returning instances of this class instead of just
    # iterators over DataItems is being able to say things like
    # `tokens.targets`, `sentences.predictions`, `documents.features`
    # and the like instead of `np.array([t.target for t in tokens])`
    # etc.

    def __init__(self, iterator):
        """Initialize sequence with given DataItem iterator."""
        self._items = list(iterator)    # TODO lazy init?

    @property
    def item_type(self):
        """Return the type of the DataItems in the sequence."""
        if not self._items:
            raise ValueError('no item type for empty sequence')
        return type(self._items[0])    # TODO enforce single type?

    @property
    def target_shape(self):
        """Return shape of DataItem target."""
        if not self._items:
            raise ValueError('no target in empty sequence')
        return self._items[0].target.shape    # TODO enforce single shape?

    @property
    def target_dim(self):
        """Return dimension of DataItem target."""
        shape = self.target_shape
        if len(shape) != 1:
            raise ValueError('target not one-dimensional')
        return shape[0]

    @property
    def targets(self):
        """Return numpy array of d.target values for DataItems d."""
        return np.array(list(d.target for d in self._items))

    @property
    def target_strs(self):
        """Return list of d.target_str values for DataItems d."""
        return [d.target_str for d in self._items]

    @property
    def predictions(self):
        """Return numpy array of d.prediction values for DataItems d."""
        return np.array(list(d.prediction for d in self._items))

    @property
    def prediction_strs(self):
        """Return list of d.prediction_str values for DataItems d."""
        return [d.prediction_str for d in self._items]

    @property
    def texts(self):
        """Return list of d.text values for DataItems d."""
        return [d.text for d in self._items]

    @property
    def features(self):
        """Return map from feature type to values for DataItems."""
        # TODO lazy init, don't assume feature types are predefined
        keys = self._items[0].feature.keys()
        return {
            f: np.array(list(e.feature[f] for e in self._items)) for f in keys
        }

    @property
    def inputs(self):
        """Return map from input type to values for DataItems."""
        if not self._items:
            return {}
        keys = self._items[0].input.keys()
        return {
            i: np.array(list(e.input[i] for e in self._items)) for i in keys
        }

    def set_target_maps(self, target_map):
        """Set mapping from target strings to values for items."""
        for item in self._items:
            item.set_target_map(target_map)

    def set_predictions(self, predictions, map_to_str=True):
        """Set prediction for items.

        Args:
            predictions: sequence of predicted values.
            map_to_str: if True, map predicted values to strings.
        """
        if len(predictions) != len(self):
            raise ValueError('prediction number mismatch')
        for item, pred in zip(self._items, predictions):
            item.set_prediction(pred)
        if map_to_str:
            self.map_predictions()

    def set_prediction_strs(self, prediction_strs):
        """Set prediction_str for items."""
        if len(prediction_strs) != len(self):
            raise ValueError('prediction number mismatch')
        for item, pred_str in zip(self._items, prediction_strs):
            item.set_prediction_str(pred_str)

    def map_predictions(self, mapper=None):
        """Map prediction to prediction_str for items."""
        if mapper is None:
            mapper = default_prediction_mapper
        for item in self._items:
            mapper(item)

    def add_feature(self, function, key=None):
        if key is None:
            key = function.name    # Use default
        if key is None:
            raise ValueError('both key and feature name are None')
        for item in self._items:
            item.add_feature(function, key)

    def add_features(self, functions):
        for function in functions:
            self.add_feature(function)

    def add_input(self, function, key=None):
        if key is None:
            key = function.key
        for item in self._items:
            item.add_input(function, key)

    def add_inputs(self, functions):
        for function in functions:
            self.add_input(function)

    def find(self, instance_of=None):
        """Return DataItemSequence of selected items.

        Args:
            instance_of: if not Null, only return nodes that are instances
                of the given class.
        """
        constraints = []
        if instance_of is not None:
            constraints.append(lambda n: isinstance(n, instance_of))
        return DataItemSequence(
            chain(*[traverse(i, yield_if=constraints) for i in self])
            )

    def __getitem__(self, key):
        """Return DataItem at given index."""
        return self._items[key]

    def __len__(self):
        """Return number of DataItems in the sequence."""
        return len(self._items)

    def __iter__(self):
        """Iterate over DataItems."""
        return iter(self._items)

class Token(TreeDataItem):
    """A single word, symbol, or other minimal element of text."""

    def __init__(self, text, target_str=None, sentence=None, position=None):
        data = { 'text': text }
        super(Token, self).__init__(
            data=data, target_str=target_str, parent=sentence,
            position=position, children=None
        )

    @property
    def text(self):
        return self.data['text']

    @property
    def sentence(self):
        return self.parent

class Sentence(TreeDataItem):
    """A sentence consisting of Tokens."""

    def __init__(self, data=None, target_str=None, document=None,
                 position=None, tokens=None):
        if data is None:
            data = {}
        super(Sentence, self).__init__(
            data=data, target_str=target_str, parent=document,
            position=position, children=tokens
        )
        if any(not isinstance(c, Token) for c in self.children):
            raise ValueError('non-Token child in Sentence')

    @property
    def document(self):
        return self.parent

    @property
    def tokens(self):
        return DataItemSequence(self.children)

class Document(TreeDataItem):
    """A document consisting of Sentences."""

    def __init__(self, target_str=None, dataset=None, position=None,
                 sentences=None):
        super(Document, self).__init__(
            data={}, target_str=target_str, parent=dataset,
            position=position, children=sentences
        )
        if any(not isinstance(c, Sentence) for c in self.children):
            raise ValueError('non-Sentence child in Document')

    @property
    def dataset(self):
        return self.parent

    @property
    def sentences(self):
        return DataItemSequence(self.children)

    @property
    def tokens(self):
        return self.find(instance_of=Token)

class Dataset(TreeDataItem):
    """A dataset consisting of Documents."""

    # TODO: this inherits target_str, target and feature from DataItem, but
    # these don't really make any sense at this level.

    def __init__(self, collection=None, position=None, documents=None,
                 name=None):
        super(Dataset, self).__init__(
            parent=collection, position=position, children=documents
        )
        if any(not isinstance(c, Document) for c in self.children):
            raise ValueError('non-Document child in Dataset')
        self.name = name

    @property
    def documents(self):
        return DataItemSequence(self.children)

    @property
    def sentences(self):
        return DataItemSequence(s for d in self.documents for s in d)

    @property
    def tokens(self):
        return self.find(instance_of=Token)

class Datasets(TreeDataItem):
    """Training, development and test Datasets."""

    # TODO: this inherits target_str, target and feature from DataItem, but
    # these don't really make any sense at this level.

    def __init__(self, train, devel, test):
        super(Datasets, self).__init__(children=[train, devel, test])
        self.train = train
        self.devel = devel
        self.test = test
        self.sets = (self.train, self.devel, self.test)
        for s in self.sets:
            s.parent = self
        self.make_targets()

    @property
    def tokens(self):
        return self.find(instance_of=Token)

    @property
    def vocabulary(self):
        return set(t.text for t in self.tokens)

    def itersets(self):
        for s in self.sets:
            yield s

    def iterdocuments(self):
        for s in self.itersets():
            for d in s.documents:
                yield d

    def itersentences(self):
        for d in self.iterdocuments():
            for s in d.sentences:
                yield s

    def itertokens(self):
        for s in self.itersentences():
            for t in s.tokens:
                yield t

    def make_targets(self):
        """Create target string to numeric maps and set maps in datasets."""
        for it, type_ in ((self.itertokens, Token),
                          (self.itersentences, Sentence),
                          (self.iterdocuments, Document)):
            target_map = make_target_map(d.target_str for d in it())
            DataItemSequence(it()).set_target_maps(target_map)

# TODO consider making this a method on one of these classes
def inverse_target_map(data, cls=Token):
    """Return mapping from index to tag string."""
    # TODO consider checking that all dataitems share the target map
    dataitems = data.find(instance_of=cls)
    if len(dataitems) == 0:
        return {}
    inverse_map = {}
    for k, v in dataitems[0].target_map.items():
        inverse_map[int(as_scalar(v))] = k
    return inverse_map

# TODO this doesn't belong here
def make_target_map(target_strs):
    """Return mapping from target strings to numeric values."""
    target_map = {}
    unique_target_strs = unique(target_strs)
    # Special case: None always maps to None (absent targets).
    include_none = False
    if None in unique_target_strs:
        unique_target_strs.remove(None)
        include_none = True
    # By convention, always map "O" to 0 (IOB-like tags).
    # TODO: check that unique_target_strs is IOB-like tagging.
    next_idx = 0
    if 'O' in unique_target_strs:
        target_map['O'] = next_idx
        unique_target_strs.remove('O')
        next_idx += 1
    for t in unique_target_strs:
        target_map[t] = next_idx
        next_idx += 1
    # Convert to one-hot
    for k in target_map:
        one_hot = np.zeros(len(target_map))
        one_hot[target_map[k]] = 1
        target_map[k] = one_hot
    if include_none:
        target_map[None] = None
    return Bidict(target_map)

# TODO this doesn't belong here
def default_prediction_mapper(item):
    one_hot_prediction = binarize_sparse(item.prediction)
    pred_str = item.target_map.inv[one_hot_prediction]
    item.set_prediction_str(pred_str)
