class Defaults(object):
    window_size = 7
    hidden_sizes = [300]
    hidden_activation = 'relu'
    max_vocab_size = 1000000
    optimizer = 'sgd' # 'adam'
    learning_rate = 0.1 # 1e-4
    epochs = 20
    iobes = True    # Map tags to IOBES on input
    max_tokens = None    # Max dataset size in tokens
    encoding = 'utf-8'    # Data encoding
    output_drop_prob = 0.0    # Dropout probablility prior to output
    token_level_eval = False    # Force token-level evaluation
    verbosity = 1    # 0=quiet, 1=progress bar, 2=one line per epoch
    fixed_wordvecs = False    # Don't fine-tune word vectors
    word_features = True
    batch_size = 50
    viterbi = True
    # Learning rate multiplier for embeddings. This is a tweak to
    # implement faster learning for embeddings compared to other
    # layers. As the feature is not yet implemented in Keras master
    # (see https://github.com/fchollet/keras/pull/1991), this option
    # currently requires the fork https://github.com/spyysalo/keras .
    embedding_lr_multiplier = 1.0
