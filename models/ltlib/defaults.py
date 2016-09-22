from logging import warn

class Defaults(object):
    encoding = 'utf-8'
    max_tokens = None
    token_level_eval = False

    def __getattr__(self, name):
        warn('missing default for {}'.format(name))
        setattr(self, name, None)
        return None

defaults = Defaults()
