import os
import sys
import argparse
from os.path import join, split, splitext
from yacs.config import CfgNode

import ltc.utils.checkpoint as cu
from ltc.config.defaults import get_cfg
import ltc.utils.misc as misc
from ltc.train_net import train
from ltc.test_net import test


def parse_args():
    parser = argparse.ArgumentParser(
        description="Provide the path to config and options. "
                    "See ltc/config/defaults.py for all options"
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Assembly101/LTContext.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See ltc/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """

    :param args: arguments including `cfg_file`, and `opts`
    :return:
        config file
    """

    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "resume_expr_num"):
        cfg.RESUME_EXPR_NUM = args.resume_expr_num
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    cfg.CONFIG_FILE = args.cfg_file
    cfg_file_name = splitext(split(args.cfg_file)[1])[0]
    cfg.OUTPUT_DIR = join(cfg.OUTPUT_DIR, cfg_file_name)

    return cfg


def prep_output_paths(cfg: CfgNode):
    """
    Preparing the path for tensorboard summary, config log and checkpoints
    :param cfg:
    :return:
    """
    if cfg.TRAIN.ENABLE:
        summary_path = misc.check_path(join(cfg.OUTPUT_DIR, "summary"))
        cfg.EXPR_NUM = misc.find_latest_experiment(join(cfg.OUTPUT_DIR, "summary")) + 1
        if cfg.TRAIN.AUTO_RESUME and cfg.TRAIN.RESUME_EXPR_NUM > 0:
            cfg.EXPR_NUM = cfg.TRAIN.RESUME_EXPR_NUM
        cfg.SUMMARY_PATH = misc.check_path(join(summary_path, "{}".format(cfg.EXPR_NUM)))
        cfg.CONFIG_LOG_PATH = misc.check_path(
            join(cfg.OUTPUT_DIR, "config", "{}".format(cfg.EXPR_NUM))
        )
        # Create the checkpoint dir.
        cu.make_checkpoint_dir(cfg.OUTPUT_DIR, cfg.EXPR_NUM)
    if cfg.TEST.ENABLE:
        os.makedirs(cfg.TEST.SAVE_RESULT_PATH, exist_ok=True)


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    prep_output_paths(cfg)
    if cfg.TRAIN.ENABLE:
        train(cfg=cfg)

    if cfg.TEST.ENABLE:
        test(cfg)


if __name__ == "__main__":
    main()
