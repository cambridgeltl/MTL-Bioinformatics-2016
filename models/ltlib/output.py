import os
import sys
import logging

from datetime import datetime
from errno import EEXIST

LOGDIR = 'logs'
PREDDIR = 'predictions'

class FileExistsError(Exception):
    pass

def main_name():
    """Return the name of the main file without the extension."""
    import __main__ as main
    return os.path.splitext(main.__file__)[0]

def safe_makedirs(path):
    """Create directory path if it doesn't already exist."""
    # From http://stackoverflow.com/a/5032238
    try:
        os.makedirs(path)
        logging.warn('Created directory %s/' % path)
    except OSError, e:
        if e.errno != EEXIST:
            raise

def _create_file(name):
    """Create new file for writing, return file object.

    Raises FileExistsError if the file exists.
    """
    # In Python < 3.3, os.open() is required for O_CREAT | O_EXCL,
    # which avoids the race conditions implicit in the naive approach.
    # Python >= 3.3 has `x` flag to `open()` (TODO: use if available).
    # (see http://stackoverflow.com/a/10979569)
    try:
        fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError, e:
        if e.errno == EEXIST:
            raise FileExistsError()
        else:
            raise
    # wrap file descriptor into file object.
    return os.fdopen(fd, 'w')

def _logname(name, dirname, suffix, time):
    """Return name like `{dirname}/{name}--{time}.{suffix}` with time in
    format 2015-12-31--23-59-59--999999.
    """
    ts = time.isoformat('_')
    ts = ts.replace(':','-').replace('_','--').replace('.','--')
    fn = '{}--{}.{}'.format(name, ts, suffix)
    return os.path.join(dirname, fn)

def logfile(name, dirname=LOGDIR, suffix='log'):
    """Return writable file object for log.

    Affixes time to name. On initial invocation, assures that the file
    does not exist.
    """
    if logfile.time is not None:
        # Reuse previously selected time.
        return open(_logname(name, dirname, suffix, logfile.time), 'w')
    while logfile.time is None:
        # Try until a time gives a file that does not exist.
        now = datetime.now()
        try:
            f = _create_file(_logname(name, dirname, suffix, now))
        except FileExistsError:
            pass
        else:
            logfile.time = now
            return f
logfile.time = None

def predfile(name=None, dirname=PREDDIR, suffix='tsv'):
    """Return writable file object for predictions."""
    if name is None:
        name = main_name()
    safe_makedirs(dirname)
    # TODO: eliminate dependency on logfile.time
    assert logfile.time is not None, 'predfile requires logging'
    return open(_logname(name, dirname, suffix, logfile.time), 'w')
        
def save_token_predictions(dataset, model, writer, vmapper=None):
    # TODO including "token" in the function spec is a bit inelegant.
    viterbi_str = '' if not vmapper else 'viterbi'
    name = '{}--{}--{}'.format(main_name(), dataset.name, viterbi_str)
    if len(dataset.tokens) > 0:
        if vmapper:
            dataset.sentences.map_predictions(vmapper)
        else:
            dataset.tokens.set_predictions(model.predict(dataset.tokens.inputs))
    with predfile(name) as out:
        writer(dataset, out)
        
def save_token_predictions_multi_output(dataset, writer, predictions=None, vmapper=None):
    # TODO including "token" in the function spec is a bit inelegant.
    viterbi_str = '' if not vmapper else 'viterbi'
    name = '{}--{}--{}'.format(main_name(), dataset.name, viterbi_str)
    assert(type(predictions).__name__ != None or vmapper != None), "Both predictions and vmapper cannot be None."
    if len(dataset.tokens) > 0:
        if vmapper:
            dataset.sentences.map_predictions(vmapper)
        if type(predictions).__name__ == 'list':
            dataset.tokens.set_predictions(predictions)
    with predfile(name) as out:
        writer(dataset, out)

def setup_logging(name=None):
    """Set up logging to stderr and file."""
    if name is None:
        name = main_name()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # logging to stderr
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(name + ':%(message)s'))
    logger.addHandler(sh)
    # logging to file
    safe_makedirs(LOGDIR)
    fh = logging.StreamHandler(logfile(name))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
    logger.addHandler(fh)
