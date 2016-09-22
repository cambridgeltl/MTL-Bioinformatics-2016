class Defaults(object):
    window_size = 7
    filter_nums = [100, 100, 100]
    filter_sizes = [3, 4, 5]
    hidden_sizes = [1200]
    hidden_activation = 'relu'
    max_vocab_size = 1000000
    optimizer = 'adam'
    learning_rate = 1e-4
    epochs = 20
    iobes = True    # Map tags to IOBES on input
    max_tokens = None    # Max dataset size in tokens
    encoding = 'utf-8'    # Data encoding
    output_drop_prob = 0.75    # Dropout probablility prior to output
    token_level_eval = False    # Force token-level evaluation
    verbosity = 1    # 0=quiet, 1=progress bar, 2=one line per epoch
    fixed_wordvecs = True    # Don't fine-tune word vectors
    word_features = False
    batch_size = 200
    train_steps = 20000 #Amount of train steps, each of batch_size to train for
    evaluate_every = 5000
    evaluate_min = 0 #The minimum step to start evaluating from
    viterbi = False
    percent_keep = 1.0 #The percentage of the given training set that should be used for training
    
class CharDefaults(object):
    word_length = 7
    filter_nums = [25, 25, 25]
    filter_sizes = [3, 4, 5]
    hidden_sizes = [20]
    hidden_activation = 'relu'
    optimizer = 'adam'
    learning_rate = 1e-4
    epochs = 20
    encoding = 'utf-8'    # Data encoding
    output_drop_prob = 0.0    # Dropout probablility prior to output
    token_level_eval = False    # Force token-level evaluation
    verbosity = 1    # 0=quiet, 1=progress bar, 2=one line per epoch
    fixed_wordvecs = True    # Don't fine-tune word vectors
    word_features = False
    batch_size = 200
    train_steps = 20000 #Amount of train steps, each of batch_size to train for
    evaluate_every = 5000
    evaluate_min = 0 #The minimum step to start evaluating from
    viterbi = False
    vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:/\|_@#$%&* +-=<>()[]{}"
    #vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,;.!?:/\|_@#$%&* +-=<>()[]{}"
