
"""Configs."""
from yacs.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

_C.TRAIN.ENABLE = True

# Dataset [breakfast, assembly101].
_C.TRAIN.DATASET = "breakfast"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 1

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 1

# batch size of eval
_C.TRAIN.EVAL_BATCH_SIZE = 1

# to use 'test' or 'val' data for evaluation
# breakfast -> test
# assembly101 -> val
_C.TRAIN.EVAL_SPLIT = "test"

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 2

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# the labels to ignore during evaluation
_C.TRAIN.EVAL_IGNORE_LABELS = []

# Experiment number to resume training from
_C.TRAIN.RESUME_EXPR_NUM = -1

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# ---------------------------------------------------------------------------- #
# Testing options.
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

_C.TEST.ENABLE = True

_C.TEST.DATASET = 'breakfast'

# the path to load trained model for testing
_C.TEST.CHECKPOINT_PATH = ""

# the labels to ignore during evaluation
_C.TEST.IGNORED_CLASSES = []

_C.TEST.BATCH_SIZE = 1

# saving the prediction
_C.TEST.SAVE_PREDICTIONS = False

_C.TEST.SAVE_RESULT_PATH = "./experiments/results"

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# model name
_C.MODEL.NAME = "ltc"

# number of classes
# breakfast: 48
# assembly101: 202
_C.MODEL.NUM_CLASSES = 48

# the dimension of extracted features, I3D: 2048
_C.MODEL.INPUT_DIM = 2048

# name of the loss, options: ['ce_mse']
_C.MODEL.LOSS_FUNC = "ce_mse"

# ignore idx for cross entropy,
# the value to be used for padding targets
_C.MODEL.PAD_IGNORE_IDX = -100

# fraction of mse loss contributes to total loss
_C.MODEL.MSE_LOSS_FRACTION = 0.15

# mse loss clamp/clip value
_C.MODEL.MSE_LOSS_CLIP_VAL = 16.0

# ----------------------------------------
# LTContext.yaml model params options
# ----------------------------------------
_C.MODEL.LTC = CfgNode()

# Number of layers
_C.MODEL.LTC.NUM_LAYERS = 9

# Size of attention window in Windowed Attention (parameter W in paper)
_C.MODEL.LTC.WINDOWED_ATTN_W = 64

# Parameter G of the Long-Term Context Attention
# which specifies the sparseness attention
# e.g. LTC_ATTN_G=1 means attention over the full sequence
_C.MODEL.LTC.LONG_TERM_ATTN_G = 64

# whether to use instance norm or not
# Our model trained with batch size of one however,
# we suggest if you want to use batch size larger than 1 USE_INSTANCE_NORM=False, to remove the InstanceNorm.
# We noticed vanilla InstanceNorm in the presence of zero padding can make the optimization unstable.
_C.MODEL.LTC.USE_INSTANCE_NORM = True

# Dilation factor of convolution, DILATION_FACTOR ** layer_idx(=0,1,2,...)
_C.MODEL.LTC.CONV_DILATION_FACTOR = 2

# number of stages
_C.MODEL.LTC.NUM_STAGES = 3

# dimension of the model
_C.MODEL.LTC.MODEL_DIM = 64

# reduced factor to reduce hidden dimension after stage 1
_C.MODEL.LTC.DIM_REDUCTION = 2.0

# the prob to mask model input
_C.MODEL.LTC.CHANNEL_MASKING_PROB = 0.3

# the prob of dropout at LTC block
_C.MODEL.LTC.DROPOUT_PROB = 0.2

# ----------------------------------------
# Transformer attention params options
# ----------------------------------------
_C.MODEL.ATTENTION = CfgNode()

_C.MODEL.ATTENTION.NUM_ATTN_HEADS = 1

# probability of dropout on attention matrix
_C.MODEL.ATTENTION.DROPOUT = 0.2

# ----------------------------------------
# MS-TCN model params options
# ----------------------------------------
_C.MODEL.TCN = CfgNode()

# Number of layers in SS_TCN and MS_TCN
_C.MODEL.TCN.NUM_LAYERS = 10

# Number of stages in SS_TCN (=1) and MS_TCN (=4)
_C.MODEL.TCN.NUM_STAGES = 4

# Number of filter in TCN
_C.MODEL.TCN.NUM_F_MAPS = 64

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = "/data/breakfast"

# the split number in cross-validation datasets:
# breakfast: 1, 2, 3, 4
_C.DATA.CV_SPLIT_NUM = 1

# background class IDs for visualization
_C.DATA.BACKGROUND_INDICES = [0]

# this is relevant for assembly101
# loading can be done from lmdb files or the saved numpy arrays
# options: ["lmdb", "numpy"]
_C.DATA.LOAD_TYPE = "lmdb"

# The fraction of videos to use for training, just used for debugging
_C.DATA.DATA_FRACTION = 1.0

# the stride to load per frame features
_C.DATA.FRAME_SAMPLING_RATE = 1

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "adam"

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-5

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 120

# Learning rate policy
# options: ['identity', 'cosine_linear_warmup', 'constant_cosine_decay']
_C.SOLVER.LR_POLICY = 'identity'

# min lr in 'cosine_linear_warmup', 'constant_cosine_decay'
_C.SOLVER.ETA_MIN = 0.00005

# Start the warm-up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR
# used in ['cosine_linear_warmup', 'constant_cosine_decay']
_C.SOLVER.WARMUP_FACTOR = 0.01

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0

# the T_max param of cosine_linear_warmup
_C.SOLVER.T_MAX = 40

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 2

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Output basedir.
_C.OUTPUT_DIR = "./experiments"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1115

# Log period in iters.
_C.LOG_PERIOD = 50

# Config file
_C.CONFIG_FILE = ""

# experiment number
_C.EXPR_NUM = -1

# tensorboard log dir
_C.SUMMARY_PATH = ""

# path to the config log file will be saved at experiment dir
_C.CONFIG_LOG_PATH = ""


def _assert_and_infer_cfg(cfg):
    import torch
    # data loader with model
    if cfg.NUM_GPUS > 0:
        gpu_count = torch.cuda.device_count()
        assert gpu_count >= cfg.NUM_GPUS,\
            f"Number of available gpus is {gpu_count} but {cfg.NUM_GPUS} is requested!"

    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
