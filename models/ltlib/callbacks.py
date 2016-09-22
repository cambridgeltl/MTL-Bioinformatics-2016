from sys import stdout
from logging import info
from datetime import datetime

from abc import ABCMeta, abstractmethod

from keras.callbacks import Callback

from evaluation import per_type_summary, conll_summary, is_iob_tagging
from util import unique
from defaults import defaults

class LtlCallback(Callback):
    """Adds after_epoch_end() to Callback.

    after_epoch_end() is invoked after all calls to on_epoch_end() and
    is intended to work around the fixed callback ordering in Keras,
    which can cause output from callbacks to mess up the progress bar
    (related: https://github.com/fchollet/keras/issues/2521).
    """

    def __init__(self):
        super(LtlCallback, self).__init__()
        self.epoch = 0

    def after_epoch_end(self, epoch):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        if epoch > 0:
            self.after_epoch_end(self.epoch)
        self.epoch += 1

    def on_train_end(self, logs={}):
        self.after_epoch_end(self.epoch)

class CallbackChain(Callback):
    """Chain of callbacks."""

    def __init__(self, callbacks):
        super(CallbackChain, self).__init__()
        self._callbacks = callbacks

    def _set_params(self, params):
        for callback in self._callbacks:
            callback._set_params(params)

    def _set_model(self, model):
        for callback in self._callbacks:
            callback._set_model(model)

    def on_epoch_begin(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_epoch_begin(*args, **kwargs)

    def on_epoch_end(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_epoch_end(*args, **kwargs)

    def on_batch_begin(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_batch_begin(*args, **kwargs)

    def on_batch_end(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_batch_end(*args, **kwargs)

    def on_train_begin(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_train_begin(*args, **kwargs)

    def on_train_end(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_train_end(*args, **kwargs)

class EvaluatorCallback(LtlCallback):
    """Abstract base class for evaluator callbacks."""

    __metaclass__ = ABCMeta

    def __init__(self, dataset, label=None, writer=None):
        super(EvaluatorCallback, self).__init__()
        if label is None:
            label = dataset.name
        if writer is None:
            writer = info
        self.dataset = dataset
        self.label = label
        self.writer = writer
        self.summaries = []

    def __call__(self):
        """Execute Callback. Invoked after end of each epoch."""
        summary = self.evaluation_summary()
        self.summaries.append(summary)
        epoch = len(self.summaries)
        for s in summary.split('\n'):
            self.writer('{} Ep: {} {}'.format(self.label, epoch, s))

    @abstractmethod
    def evaluation_summary(self):
        """Return string summarizing evaluation results."""
        pass

    def after_epoch_end(self, epoch):
        self()

class EpochTimer(LtlCallback):
    """Callback that logs timing information."""

    def __init__(self, label='', writer=info):
        super(EpochTimer, self).__init__()
        self.label = '' if not label else label + ' '
        self.writer = writer

    def on_epoch_begin(self, epoch, logs={}):
        super(EpochTimer, self).on_epoch_begin(epoch, logs)
        self.start_time = datetime.now()

    def after_epoch_end(self, epoch):
        end_time = datetime.now()
        delta = end_time - self.start_time
        start = str(self.start_time).split('.')[0]
        end = str(end_time).split('.')[0]
        self.writer('{}Ep: {} {}s (start {}, end {})'.format(
                self.label, epoch, delta.seconds, start, end
                ))

class Predictor(LtlCallback):
    """Makes and stores predictions for data item sequence."""

    def __init__(self, dataitems):
        super(Predictor, self).__init__()
        self.dataitems = dataitems

    def after_epoch_end(self, epoch):
        predictions = self.model.predict(self.dataitems.inputs)
        self.dataitems.set_predictions(predictions)

class PredictionMapper(LtlCallback):
    """Maps predictions to strings for data item sequence."""

    def __init__(self, dataitems, mapper):
        super(PredictionMapper, self).__init__()
        self.dataitems = dataitems
        self.mapper = mapper

    def after_epoch_end(self, epoch):
        self.dataitems.map_predictions(self.mapper)
        # TODO check if summary() is defined
        info(self.mapper.summary())

class ConllEvaluator(EvaluatorCallback):
    """Evaluates performance using CoNLL criteria."""

    def __init__(self, dataset, label=None, writer=None):
        super(ConllEvaluator, self).__init__(dataset, label, writer)

    def evaluation_summary(self):
        return conll_summary(self.dataset.sentences)

class TokenLevelEvaluator(EvaluatorCallback):
    """Evaluates performance using token-level metrics."""

    def __init__(self, dataset, label=None, writer=None):
        super(TokenLevelEvaluator, self).__init__(dataset, label, writer)

    def evaluation_summary(self):
        return per_type_summary(self.dataset.tokens)

class TokenAccuracyEvaluator(EvaluatorCallback):
    """Evaluates performance using token-level accuracy."""

    # TODO why does this class exist? Isn't TokenLevelEvaluator better
    # in every way?

    def __init__(self, dataset, label=None, writer=None):
        super(TokenAccuracyEvaluator, self).__init__(dataset, label, writer)

    def evaluation_summary(self):
        gold = self.dataset.tokens.target_strs
        pred = self.dataset.tokens.prediction_strs
        assert len(gold) == len(pred)
        total = len(gold)
        correct = sum(int(p==g) for p, g in zip(pred, gold))
        return 'acc: {:.2%} ({}/{})'.format(1.*correct/total, correct, total)

def token_evaluator(dataset, label=None, writer=None, mapper=None,
                    config=defaults):
    """Return appropriate evaluator callback for dataset."""
    if config.token_level_eval:
        evaluator = TokenLevelEvaluator
    elif is_iob_tagging(unique(dataset.tokens.target_strs)):
        evaluator = ConllEvaluator
    else:
        evaluator = TokenLevelEvaluator    # default
    info('using {} for {}'.format(evaluator.__name__, dataset.name))

    callbacks = []
    callbacks.append(Predictor(dataset.tokens))
    callbacks.append(evaluator(dataset, label=label, writer=writer))
    if mapper is not None:
        # TODO don't assume the mapper expects sentences.
        callbacks.append(PredictionMapper(dataset.sentences, mapper))
        # TODO do we really want a second eval here?
        callbacks.append(evaluator(dataset, label=label, writer=writer))
    return CallbackChain(callbacks)
