import inspect
import argparse
import re

from logging import info, warn
from json import dumps
from hashlib import md5

class Settings(object):
    def __init__(self, values=None):
        """Return Settings with given values."""
        if isinstance(values, object):
            values = vars(values)
        elif not isinstance(values, dict):
            raise NotImplementedError(type(values).__name__)
        self.__dict__.update(**values)

    def __getattr__(self, name):
        if _is_special(name):
            raise AttributeError
        else:
            warn('setting {} not defined'.format(name))
            setattr(self, name, None)
            return None

    # Note: the following are not methods on Settings to avoid
    # conflicting with possible parameter names.

def cli_settings(positional, optional, argv=None):
    """Return Settings from CLI arguments."""
    if argv is None:
        import sys
        argv = sys.argv
    parser = _argparser(positional, optional)
    args = parser.parse_args(argv[1:])
    settings = Settings(args)
    log_settings(settings)    # TODO make optional?
    return settings

def log_settings(settings, logger=info):
    """Log settings with logging function logger."""
    # Record also checksum for settings without datadir to make it
    # easier to identify logs for same settings and different data
    checksums = _checksum(settings) + '/' + _checksum(settings, ['datadir'])
    logger('md5:'+checksums+'\n'+_serialize(settings))

def _serialize(settings, exclude=None):
    """Return a consistent, human-readable string serialization of settings."""
    if exclude is None:
        exclude = []
    sdict = dict(_variables(settings))
    sdict = { k: v for k, v in sdict.items() if k not in exclude }
    sstr = dumps(sdict, sort_keys=True, indent=4, separators=(',', ': '))
    # Remove linebreaks from values (nicer layout)
    while True:
        snew = re.sub(r'\n\s*([^"\s\}])', r' \1', sstr)
        if snew == sstr:
            break
        sstr = snew
    return sstr

def _checksum(settings, exclude=None):
    """Return checksum for settings."""
    sstr = _serialize(settings, exclude)
    return md5(sstr).hexdigest()

def _argparser(positional, optional):
    """Return ArgumentParser taking given positional and optional arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add = parser.add_argument
    for arg in positional:
        add(arg)
    # take optional args and defaults from arguments class variables and values
    for arg, val in _variables(optional):
        name, help_ = arg.replace('_', '-'), arg.replace('_', ' ')
        type_ = type(val) if val is not None else int    # assume int for None
        if type_ in (int, float, str):
            add('--'+name, metavar=_typename(val), type=type_, default=val,
                help=help_)
        elif type_ is list:
            add('--'+name, metavar=_typename(val), type=type(val[0]),
                nargs='+', default=val, help=help_)
        elif type_ is bool:
            name = 'no-'+name if val else name
            act = 'store_' + ('false' if val else 'true')
            add('--'+name, dest=arg, default=val, action=act,
                help='toggle ' + help_)
        else:
            raise NotImplementedError(type_.__name__)
    return parser

def _is_special(name):
    """Return True if method name is special, False otherwise."""
    return name.startswith('__') and name.endswith('__')

def _variables(cls, include_special=False):
    """Return class variables."""
    variables = inspect.getmembers(cls, lambda m: not inspect.isroutine(m))
    if not include_special:
        variables = [v for v in variables if not _is_special(v[0])]
    return variables

def _typename(d):
    d = d[0] if type(d) is list else d
    t = type(d) if d is not None else int
    return t.__name__.upper()
