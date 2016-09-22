from keras import optimizers

DEFAULT_OPTIMIZER = 'adam'

def get_optimizer(config):
    """Return optimizer specified by configuration."""
    config = vars(config)
    name = config.get('optimizer', DEFAULT_OPTIMIZER)
    optimizer = optimizers.get(name)    # Default parameters
    lr = config.get('learning_rate')
    if lr is not None:
        optimizer = type(optimizer)(lr=lr)
    return optimizer
