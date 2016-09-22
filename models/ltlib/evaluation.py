import numpy as np    # TODO remove dependency

import conlleval

from collections import namedtuple
from itertools import chain

from util import unique

from logging import warn

BinaryClassificationCounts = namedtuple('BinaryClassificationCounts',
                                        'tp tn fp fn')
BinaryClassificationMetrics = namedtuple('BinaryClassificationMetrics',
                                         'tp tn fp fn acc prec rec fscore')

# Tag for "out" in tagging scheme (IOB, IOBES, etc.)
OUT_TAG = 'O'

def accuracy(gold, pred):
    if len(gold) != len(pred):
        raise ValueError('count mismatch')
    correct = sum(int(g == p) for g, p in zip(gold, pred))
    return 1.*correct/len(gold)

def tp_tn_fp_fn(gold, pred):
    """Return (TP, FN, FP, FN) counts for gold and prediced values.

    Assumes that 0 is negative and all others positive.
    """
    tp, tn, fp, fn = 0, 0, 0, 0
    for g, p in zip(gold, pred):
        if g == p:
            if g == 0:
                tn += 1
            else:
                tp += 1
        else:
            if g == 0:
                fp += 1
            else:
                fn += 1
    return BinaryClassificationCounts(tp, tn, fp, fn)

def precision_recall_fscore(tp, fp, fn):
    """Return (precision, recall, f-score) for given counts."""
    prec = 0.0 if tp + fp == 0 else 1.*tp / (tp + fp)
    rec = 0.0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0.0 if prec + rec == 0.0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f

def evaluate_binary_classification(gold, pred, positive):
    """Evaluate binary classification performance.

    Map labels in positive to 1 and others to 0.

    Return BinaryClassificationMetrics.
    """
    if len(gold) != len(pred):
        raise ValueError('count mismatch')

    gold = _binarize(gold, positive)
    pred = _binarize(pred, positive)

    if not any(i for i in gold):
        warn('no positive gold labels for %s' % str(positive))

    acc = accuracy(gold, pred)
    tp, tn, fp, fn = tp_tn_fp_fn(gold, pred)
    prec, rec, f = precision_recall_fscore(tp, fp, fn)

    return BinaryClassificationMetrics(tp, tn, fp, fn, acc, prec, rec, f)

# TODO _tag_type, _iob_types and is_iob_tagging would make more sense
# somewhere else.

def _tag_type(tag):
    """Return type name for IOB-like tag or None if out or not an IOB tag."""
    if tag == OUT_TAG or tag[:2] not in ('B-', 'I-', 'E-', 'S-'):
        return None
    else:
        return tag[2:]

def _iob_types(tags):
    """Return type names for IOB-like labels."""
    return set(_tag_type(t) for t in tags if t != OUT_TAG)

def is_iob_tagging(tags):
    """Return True if given tags are an IOB-like tagging, False otherwise."""
    if OUT_TAG not in tags:
        return False
    else:
        return all(_tag_type(t) for t in tags if t != OUT_TAG)

def _group_tags(tags):
    """Group tag strings into sets of equivalents.

    For IOB-like tags groups together "in" tags (B-, I-, etc.) by type.

    Return list of (group-name, positive-tags) tuples.
    """
    if not is_iob_tagging(tags):
        return [(t, [t]) for t in tags]    # no groups
    else:
        types = _iob_types(tags)
        return [(g, [t for t in tags if t.endswith('-'+g)]) for g in types]

def _binarize(a, positive):
    """Return values mapped to 1 or 0.

    Map values in positive to 1 and others to 0.
    """
    return [1 if i in positive else 0 for i in a]

def average_precision_recall_fscore(results, micro=True):
    """Return average precision, recall and f-score for list of
    BinaryClassificationMetrics.
    """
    if micro:
        total = BinaryClassificationMetrics(*tuple(np.sum(results, axis=0)))
        return precision_recall_fscore(total.tp, total.fp, total.fn)
    else:
        avg = BinaryClassificationMetrics(*tuple(np.average(results, axis=0)))
        return avg.prec, avg.rec, avg.fscore

def per_type_summary(dataitems):
    """Return string summarizing per-class classification performance."""
    gold = dataitems.target_strs
    pred = dataitems.prediction_strs
    tags = unique(chain(gold, pred))
    by_type = {}
    for name, positive in _group_tags(tags):
        by_type[name] = evaluate_binary_classification(gold, pred, positive)
    acc = accuracy(gold, pred)
    _, _, micf = average_precision_recall_fscore(by_type.values(), micro=True)
    _, _, macf = average_precision_recall_fscore(by_type.values(), micro=False)
    summaries = [
        'acc: {:.2%} micf: {:.2%} macf: {:.2%}'.format(acc, micf, macf)
    ]
    nlen = max(len(name) for name, _ in _group_tags(tags))
    for name, r in sorted(by_type.items()):
        summaries.append((
            '{name:{nlen}} f: {m.fscore:.2%} ' +
            '(p:{m.prec:.1%} r:{m.rec:.1%} tp:{m.tp} fp:{m.fp} fn:{m.fn})'
        ).format(name=name, nlen=nlen, m=r))
    return '\n'.join(summaries)

def conll_summary(sentences):
    eval_sentences = [
        [(t.target_str, t.prediction_str) for t in s] for s in sentences
    ]
    gold = [t.target_str for s in sentences for t in s]
    pred = [t.prediction_str for s in sentences for t in s]
    acc = accuracy(gold, pred)
    counts = conlleval.evaluate_sentences(eval_sentences)
    overall, by_type = conlleval.metrics(counts)
    #print("By type keys: ", len(by_type.keys()))
    #nlen = max(len(name) for name in by_type.keys()) if len(by_type.keys()) > 0 else 0
    nlen = max(len(name) for name in by_type.keys())

    summaries = [(
        'acc: {acc:.2%} f: {m.fscore:.2%} ' +
        '(p:{m.prec:.1%} r:{m.rec:.1%} tp:{m.tp} fp:{m.fp} fn:{m.fn})'
    ).format(acc=acc, m=overall)]
    for name, r in sorted(by_type.items()):
        summaries.append((
            '{name:{nlen}} f: {m.fscore:.2%} ' +
            '(p:{m.prec:.1%} r:{m.rec:.1%} tp:{m.tp} fp:{m.fp} fn:{m.fn})'
        ).format(name=name, nlen=nlen, m=r))
    return '\n'.join(summaries)
