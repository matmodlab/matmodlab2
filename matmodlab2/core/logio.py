import logging
from ..version import VERSION
from .environ import environ

SPLASH = r"""
 _____ ______   _____ ______   ___        _______
|\   _ \  _   \|\   _ \  _   \|\  \      /  ___  \
\ \  \\\__\ \  \ \  \\\__\ \  \ \  \    /__/|_/  /|
 \ \  \\|__| \  \ \  \\|__| \  \ \  \   |__|//  / /
  \ \  \    \ \  \ \  \    \ \  \ \  \____  /  /_/__
   \ \__\    \ \__\ \__\    \ \__\ \_______\\________\
    \|__|     \|__|\|__|     \|__|\|_______|\|_______|
                   Material Model Laboratory2 v {0}

""".format(
    ".".join("{0}".format(i) for i in VERSION)
)

# Monkey path the logging stream handler emit function
logging.basicConfig(format="%(message)s")


def emit(self, record):
    """Monkey-patch the logging StreamHandler emit function. Allows omiting
    trailing newline when not wanted"""
    if hasattr(self, "baseFilename"):
        fs = "%s\n"
    else:
        fs = "%s" if getattr(record, "continued", False) else "%s\n"
    self.stream.write(fs % self.format(record))
    self.flush()


logging.StreamHandler.emit = emit


def get_logger(name, verbosity=None):
    """Set up the logger"""

    if environ.notebook:
        level = logging.WARNING

    elif environ.parent_process:
        level = logging.CRITICAL

    elif verbosity is not None:
        environ.verbosity = verbosity
        level = environ.loglevel
    else:
        level = environ.loglevel

    logger = logging.getLogger(name)
    logger.propagate = False
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)

    return logger


def add_filehandler(logger, filename):
    for handler in logger.handlers:
        if hasattr(handler, "baseFilename"):
            logger.removeHandler(handler)
    fh = logging.FileHandler(filename, mode="w")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)


def splash(logger):
    splashed = getattr(logger, "splashed", False)
    if not splashed:
        logger.info(SPLASH)
        logger.splashed = True


class FortranError(Exception):
    pass


def StopFortran(message):
    raise FortranError(message)


logger = get_logger("matmodlab")
