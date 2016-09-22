import numpy as np

from math import log
from collections import defaultdict
from logging import info

from data import Token, inverse_target_map
from util import as_scalar

def transition_probabilities(sequences, smoothing=None, binarize=True,
                             log_prob=True, verbose=False):
    """Estimate start and transition probabilities from sequences.

    Args:
        sequences: Sequences of DataItems.
        smoothing: Type of smoothing to apply, or None for none.
        binarize: Whether to convert probabilities to 0.0 or 1.0.
        log_prob: Whether to return log probabilities.
        verbose: Whether to write out statistics.

    Return:
        arrays (sprob, tprob) where sprob[i] is the probability of i at
        sequence start and tprob[j][i] the probability of i followed by j
        (note i, j inversion).
    """
    counts = get_counts(sequences)
    if smoothing is not None:
        counts = smooth_counts(counts, smoothing=smoothing)
    stprob = counts_to_probabilities(counts)
    if binarize:
        stprob = binarize_probabilities(stprob)
    if log_prob:
        stprob = log_probabilities(stprob)
    if verbose:
        report_statistics(counts, stprob)
    # transpose tprob for nicer inner loop in viterbi()
    return stprob[0], stprob[1].transpose()

def emission_probabilities(dataitems, log_prob=True):
    """Return emission probabilities for dataitems.

    Args:
        dataitems: Sequence of DataItems.
        log_prob: Whether to return log probabilities.

    Return:
        array prob where prob[i][j] is the probability of the
        observation at i in from state j.
    """
    eprob = np.array([ i.prediction for i in dataitems ])
    if log_prob:
        with np.errstate(divide='ignore'):    # log(0) = -inf OK
            eprob = np.log(eprob)
    return eprob

def viterbi(states, stprob, eprob):
    sprob, tprob = stprob
    # eprob[i][j] is included in sums for prob[i][j] for all i, j, so
    # start with a copy and skip the addition.
    prob = eprob.copy()
    prob[0] += sprob
    for i in range(1, len(eprob)):
        for j in range(len(states)):
            prob[i][j] += max(prob[i-1] + tprob[j])
    return [states[np.argmax(prob[i])] for i in range(len(eprob))]

def _target_size(sequences):
    """Return the target size of the items in the sequences."""
    # TODO consider checking consistency of target sizes.
    for sequence in sequences:
        for item in sequence:
            return len(item.target)
    return None

def get_counts(sequences):
    """Return start and transition counts from sequences.

    Returns:
        arrays (scount, tcount) where scount[i] is the count of i at
        sequence start and tcount[i][j] the count of i followed by j.
    """
    size = _target_size(sequences)
    assert size is not None, 'internal error'
    scount = np.zeros(size, dtype='int')
    tcount = np.zeros((size, size), dtype='int')
    for sequence in sequences:
        i = None
        for item in sequence:
            j = as_scalar(item.target)
            if i is None:
                scount[j] += 1
            else:
                tcount[i][j] += 1
            i = j
    return scount, tcount

def smooth_counts(counts, smoothing):
    """Apply smoothing to inital and transition counts.

    Args:
        counts: arrays (scount, tcount) where scount[i] is the count of i
            at sequence start and tcount[i][j] the count of i followed by j.

    Returns:
        arrays (scount, tcount) with smoothed counts.
    """
    scount, tcount = counts
    if smoothing == 'add-one':
        for j in range(len(scount)):
            scount += 1
        for i in range(len(tcount)):
            for j in range(len(tcount[i])):
                tcount[i][j] += 1
    elif smoothing is not None:
        raise ValueError('smoothing {}'.format(smoothing))
    return scount, tcount

def counts_to_probabilities(counts):
    """Return start and transition probabilities for given counts.

    Args:
        counts: arrays (scount, tcount) where scount[i] is the count of i
            at sequence start and tcount[i][j] the count of i followed by j.

    Return:
        arrays (sprob, tprob) where sprob[i] is the probability of i at
        sequence start and tprob[i][j] the probability of i followed by j.
    """
    scount, tcount = counts
    sprob = scount.astype('float') / scount.sum()
    tprob = np.zeros(tcount.shape, dtype='float')
    # TODO do this in numpy
    for i in range(len(tcount)):
        t = tcount[i].sum()
        if t == 0:
            for j in range(len(tcount[i])):
                tprob[i][j] = 0.0
        else:
            for j in range(len(tcount[i])):
                tprob[i][j] = 1.*tcount[i][j] / t
    return sprob, tprob

def binarize_probabilities(stprob, threshold=0.0):
    """Map probabilities > threshold to 1.0 and others to 0.0."""
    sprob, tprob = stprob
    sprob = (sprob > threshold).astype('float')
    tprob = (tprob > threshold).astype('float')
    return sprob, tprob

def log_probabilities(stprob):
    """Take log of probabilities."""
    sprob, tprob = stprob
    with np.errstate(divide='ignore'):    # log(0) = -inf OK
        sprob = np.log(sprob)
        tprob = np.log(tprob)
    return sprob, tprob

def report_statistics(counts, stprob, writer=info):
    """Write out count and probability statistics."""
    scount, tcount = counts
    sprob, tprob = stprob
    for i in range(len(scount)):
        writer('START {}: {}/{} ({:.2%})'.format(i, scount[i], scount.sum(),
                                                 sprob[i]))
    for i in range(len(tcount)):
        for j in range(len(tcount[i])):
            t = tcount[i].sum()
            writer('{} -> {}: {}/{} ({:.2%})'.format(
                    i, j, tcount[i][j], t, tprob[i][j]))

# TODO consider moving this, it's not part of the core functionality here.
class TokenPredictionMapper(object):
    """Maps predictions for Tokens to strings using Viterbi decoding."""

    def __init__(self, sentences, smoothing=None, binarize=True):
        self.tprob = transition_probabilities(sentences, smoothing, binarize)
        # TODO: get rid of index_to_state and states
        self.index_to_state = inverse_target_map(sentences, cls=Token)
        self.states = self.index_to_state.values()
        self.total_count = 0
        self.changed_count = 0

    def _count_changes(self, tokens, mapped):
        """Take statistics for reporting."""
        original = tokens.prediction_strs
        changed = sum(int(m != o) for m, o in zip(mapped, original))
        self.total_count += len(original)
        self.changed_count += changed

    def summary(self):
        """Return string summarizing statistics."""
        summary = 'updated {:.4%} ({}/{})'.format(
            1.*self.changed_count/self.total_count,
            self.changed_count, self.total_count
            )
        self.changed_count = 0
        self.total_count = 0
        return summary

    def __call__(self, sentence):
        tokens = sentence.tokens
        eprob = emission_probabilities(tokens)
        mapped = viterbi(self.states, self.tprob, eprob)
        self._count_changes(tokens, mapped)
        tokens.set_prediction_strs(mapped)

def get_prediction_mapper(dataitems, config):
    if config.viterbi:
        return TokenPredictionMapper(dataitems)
    else:
        return None
