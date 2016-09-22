import io
import sys

from os import path
from logging import warn

from util import lookaround
from data import Token, Sentence, Document, Dataset, Datasets
from data import inverse_target_map

from defaults import defaults

DOCUMENT_SEPARATOR = '-DOCSTART-'

class FormatError(Exception):
    pass

def load(filename, config=defaults, name=None):
    """Load CoNLL-style file, return Dataset."""
    if name is None:
        name = path.splitext(path.basename(filename))[0]
    with io.open(filename, encoding=config.encoding) as f:
        return read(f, config, name)

def load_dir(directory, config=defaults):
    """Load train, devel and test data from directory, return Datasets."""
    # Datasets are assumed to be named {train,devel,test}.tsv.
    datasets = []
    for dset in ('train', 'devel', 'test'):
        name = '{}-{}'.format(path.basename(directory.rstrip('/')), dset)
        datasets.append(load(path.join(directory, dset+'.tsv'), config, name))
    return Datasets(*datasets)

class ParseState(object):
    """Support object for read()."""

    def __init__(self, config=defaults, name=None):
        self.config = config
        self.dataset = Dataset(name=name)
        self.document = Document()
        self.texts = []
        self.tags = []

    def sentence_break(self):
        if len(self.texts) == 0:
            return
        if self.config.iobes:
            self.tags = iob_to_iobes(self.tags)
        tokens = [Token(t, g) for t, g in zip(self.texts, self.tags)]
        self.document.add_child(Sentence(tokens=tokens))
        self.texts = []
        self.tags = []

    def document_break(self):
        self.sentence_break()
        if len(self.document) == 0:
            return
        self.dataset.add_child(self.document)
        self.document = Document()

    def finish(self):
        self.document_break()

def read(flo, config=defaults, name=None):
    """Read CoNLL-style data from file-like object, return Dataset."""
    state = ParseState(config, name)
    token_num = 0
    for ln, line in enumerate(flo, start=1):
        line = line.rstrip('\r\n')
        if _is_document_separator(line):
            state.document_break()
        elif _is_sentence_separator(line):
            state.sentence_break()
        else:
            fields = line.split()
            if len(fields) != 2:
                raise FormatError('line {}: {}'.format(ln, line))
            state.texts.append(fields[0])
            state.tags.append(fields[-1])
            token_num += 1
        if config.max_tokens is not None and token_num >= config.max_tokens:
            warn('stopping reading {} at max_tokens ({})'.format(
                flo.name, config.max_tokens
            ))
            break
    state.finish()
    return state.dataset

def write(dataset, flo=None):
    """Write CoNLL-style data to file-like object.

    The format is that expected by the conlleval script, with lines
    containing token-text, correct-tag, and guessed-tag separated by
    space, sentences separated by empty lines, and documents by a
    separator token.
    """
    if flo is None:
        flo = sys.stdout
    for document in dataset.documents:
        flo.write('{} O O\n\n'.format(DOCUMENT_SEPARATOR))
        for sentence in document.sentences:
            for t in sentence.tokens:
                flo.write('{} {} {}\n'.format(t.text, t.target_str,
                                              t.prediction_str))
            flo.write('\n')

def write_probabilities(dataset, flo=None):
    """Write data with predicted probabilities to file-like object."""
    if flo is None:
        flo = sys.stdout
    for document in dataset.documents:
        flo.write('{} O O\n\n'.format(DOCUMENT_SEPARATOR))
        for sentence in document.sentences:
            tokens = sentence.tokens
            idx_to_str = inverse_target_map(tokens)
            for t in tokens:
                probs = ' '.join('{}:{}'.format(idx_to_str[i], p)
                                 for i, p in enumerate(t.prediction))
                flo.write('{} {} {}\n'.format(t.text, t.target_str, probs))
            flo.write('\n')

def _is_document_separator(line):
    """Return True if line is a CoNLL document separator, False otherwise."""
    fields = line.split()
    return fields and fields[0] == DOCUMENT_SEPARATOR

def _is_sentence_separator(line):
    """Return True if line is a CoNLL sentence separator, False otherwise."""
    return line.strip() == ''

# map pattern of (previous, current, next) IOB tags to IOBES
_iobes_tag = {
    ('B', 'I', 'B'): 'E',
    ('B', 'I', 'I'): 'I',
    ('B', 'I', 'O'): 'E',
    ('I', 'I', 'B'): 'E',
    ('I', 'I', 'I'): 'I',
    ('I', 'I', 'O'): 'E',
    ('O', 'I', 'B'): 'S',
    ('O', 'I', 'I'): 'B',
    ('O', 'I', 'O'): 'S',
    ('B', 'B', 'B'): 'S',
    ('B', 'B', 'I'): 'B',
    ('B', 'B', 'O'): 'S',
    ('I', 'B', 'B'): 'S',
    ('I', 'B', 'I'): 'B',
    ('I', 'B', 'O'): 'S',
    ('O', 'B', 'B'): 'S',
    ('O', 'B', 'I'): 'B',
    ('O', 'B', 'O'): 'S',
}

def iob_to_iobes(iob_tags):
    """Map sequence of IOB tags to IOBES."""
    iobes_tags = []
    def tag(t):
        return 'O' if t is None else t[0]
    def ttype(t):
        return None if t in (None, 'O') else t[1:]
    for i, (prev, curr, next_) in enumerate(lookaround(iob_tags[:])):
        if curr is None:
            break
        if tag(curr) == 'O':
            t = 'O'    # no change to out tags
        else:
            prev_tag, curr_tag, next_tag = tag(prev), tag(curr), tag(next_)
            # For mapping, previous/next tags not matching the current in
            # type are considered out so that e.g. (O, I-LOC, I-PER)
            # maps like (O, I-LOC, O).
            if ttype(prev) != ttype(curr):
                prev_tag = 'O'
            if ttype(next_) != ttype(curr):
                next_tag = 'O'
            try:
                t = _iobes_tag[prev_tag, curr_tag, next_tag] + curr[1:]
            except KeyError:
                warn('failed to map tag {} to IOBES'.format(curr))
                t = curr
        iobes_tags.append(t)
    return iobes_tags
