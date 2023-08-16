from os import makedirs
from os.path import join
import decimal
import logging
import sys

import simplejson
import pandas as pd


def setup_logging(output_dir, expr_num=0, file_prefix="expr", file_mode="a"):
    """
        Sets up the logging
    :param output_dir:
    :param expr_num:
    :param file_prefix:
    :param file_mode:
    :return:
    """
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
    log_folder = join(output_dir, "logs")
    makedirs(log_folder, exist_ok=True)
    logfile_name = f"{file_prefix}_{expr_num:03d}.log"
    log_file = join(log_folder, logfile_name)

    fh = logging.FileHandler(filename=log_file, mode=file_mode)
    ch = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, format=_FORMAT, handlers=[fh, ch])
    return log_file


def get_logger(name):
    """
        Retrieve the logger with the specified name or, if name is None, return a
        logger which is the root logger of the hierarchy.
    :param name: name of the logger.
    :return:
    """
    return logging.getLogger(name)


def log_json_stats(stats, precision=4):
    """
    Logs json stats.
    :param stats: a dictionary or Dataframe of statistical information to log.
    :param precision: number decimal points for float values
    :return:
    """
    if isinstance(stats, pd.DataFrame):
        stats = stats.to_dict()
    stats = {
        k: decimal.Decimal("{:.{perc}f}".format(v, perc=precision)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    logger = get_logger(__name__)
    logger.info("json_stats: {:s}".format(json_stats))
