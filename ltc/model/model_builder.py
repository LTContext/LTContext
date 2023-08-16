from yacs.config import CfgNode

import torch
from ltc.model.ms_tcn import MultiStageModel
from ltc.model.ltcontext import LTC

# Supported models
_MODEL_TYPES = {
    "ms-tcn": MultiStageModel,
    "ltc": LTC,
}


def build_model(cfg: CfgNode) -> torch.nn.Module:
    """
     Builds the action segmentation model.

    :param cfg: configs that contains the parameters to build the model.
                Details can be seen in ltc/config/defaults.py.
    :return:
        model
    """
    assert (
        cfg.MODEL.NAME in _MODEL_TYPES.keys()
    ), "Model type '{}' not supported".format(cfg.MODEL.NAME)
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    # Construct the model
    model = _MODEL_TYPES[cfg.MODEL.NAME](cfg.MODEL)

    return model
