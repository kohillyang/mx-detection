# encoding=utf-8
def lsdir(rootdir="", suffix=".png"):
    import os
    assert os.path.exists(rootdir)
    for r, y, names in os.walk(rootdir):
        for name in names:
            if unicode(name).endswith(suffix):
                yield os.path.join(r, name)


def log_init(filename):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    for h in logger.handlers:
        logger.removeHandler(h)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)s][line: %(lineno)d][%(levelname)s] ## %(message)s')
    fh.setFormatter(formatter)
    formatter = logging.Formatter(
        '[%(filename)s][line: %(lineno)d][%(levelname)s] ## %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
