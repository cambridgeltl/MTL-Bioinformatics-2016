import numpy as np

from keras import backend as K
from keras import initializations
from keras.layers import Layer, Input, Embedding, merge

class FixedEmbedding(Layer):
    """Embedding with fixed weights.

    Modified from keras/layers/embeddings.py in Keras (http://keras.io).

    WARNING: this is experimental and not fully tested, use at your
    own risk.
    """
    input_ndim = 2

    def __init__(self, input_dim, output_dim, weights=None, input_length=None,
                 mask_zero=False, dropout=0., **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.mask_zero = mask_zero
        self.dropout = dropout

        if 0. < self.dropout < 1.:
            self.uses_learning_phase = True

        if (not isinstance(weights, list) or len(weights) != 1 or
            weights[0].shape != (input_dim, output_dim)):
            raise ValueError('weights must be a list with single element'
                             ' with shape (input_dim, output_dim).')
        self.initial_weights = weights

        kwargs['input_shape'] = (self.input_length,)
        kwargs['input_dtype'] = 'int32'
        super(FixedEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = K.variable(np.zeros((self.input_dim, self.output_dim),
                                     dtype='int32'),
                            name='{}_W'.format(self.name))
        self.non_trainable_weights = [self.W]
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)

    def get_output_shape_for(self, input_shape):
        if not self.input_length:
            input_length = input_shape[1]
        else:
            input_length = self.input_length
        return (input_shape[0], input_length, self.output_dim)

    def call(self, x, mask=None):
        if 0. < self.dropout < 1.:
            retain_p = 1. - self.dropout
            B = K.random_binomial((self.input_dim,), p=retain_p) * (1. / retain_p)
            B = K.expand_dims(B)
            W = K.in_train_phase(self.W * B, self.W)
        else:
            W = self.W
        out = K.gather(W, x)
        return out

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'input_length': self.input_length,
                  'mask_zero': self.mask_zero,
                  'dropout': self.dropout}
        base_config = super(Embedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def input_and_embedding(embedding, input_length, name=None, fixed=False,
                        **kwargs):
    """Create Input layer followed by embedding."""
    if name is None:
        name = embedding.name
    i = Input(shape=(input_length,), dtype='int32', name=name)
    E = Embedding if not fixed else FixedEmbedding
    e = E(embedding.input_dim, embedding.output_dim,
          weights=[embedding.weights], input_length=input_length,
          **kwargs)(i)
    return i, e

def inputs_and_embeddings(features, config):
    inputs, embeddings = [], []
    window, fixed, args = config.window_size, config.fixed_wordvecs, {}
    if (config.embedding_lr_multiplier and
        config.embedding_lr_multiplier != 1.0):
        # Note: the per-layer learning rate multiplier argument
        # `W_lr_multiplier` is not supported in Keras master
        # (see https://github.com/fchollet/keras/pull/1991).
        # Grab the fork fork https://github.com/spyysalo/keras
        # to use this option.
        args['W_lr_multiplier'] = config.embedding_lr_multiplier
    for f in features:
        kwargs = args.copy()
        if fixed:
            # No learning rate multiplier in fixed embedding
            kwargs.pop('W_lr_multiplier', None)
        i, e = input_and_embedding(f, window, fixed=fixed, **kwargs)
        inputs.append(i)
        embeddings.append(e)
        # By convention, word vectors are the first (index 0) feature.
        # No other embedding features can be fixed.
        # TODO: generalize identification of word vectors.
        fixed = False
    return inputs, embeddings

def concat(inputs, concat_axis=-1, output_shape=None, name=None):
    """Concatenate tensors.

    This is Keras merge with mode='concat' and support for the
    degenerate case of catenating a single input.
    """
    if len(inputs) == 1:
        # Degenerate case. TODO: handle output_shape and name.
        return inputs[0]
    else:
        return merge(inputs, mode='concat', concat_axis=concat_axis,
                     output_shape=output_shape, name=name)
