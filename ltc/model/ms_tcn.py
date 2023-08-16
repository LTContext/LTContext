# code taken from https://github.com/yabufarha/ms-tcn
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiStageModel(nn.Module):
    def __init__(self, model_cfg):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(model_cfg.TCN.NUM_LAYERS,
                                       model_cfg.TCN.NUM_F_MAPS,
                                       model_cfg.INPUT_DIM,
                                       model_cfg.NUM_CLASSES)
        self.stages = nn.ModuleList([
            copy.deepcopy(SingleStageModel(model_cfg.TCN.NUM_LAYERS,
                                           model_cfg.TCN.NUM_F_MAPS,
                                           model_cfg.NUM_CLASSES,
                                           model_cfg.NUM_CLASSES)) for s in range(model_cfg.TCN.NUM_STAGES - 1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, masks):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, masks)
        out = self.conv_out(out) * masks[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x, masks):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        return (x + out) * masks


