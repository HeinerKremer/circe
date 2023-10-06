import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
from scipy.linalg import solve as scp_solve
from cmr.methods.fgel_neural import NeuralFGEL

from trainer.base_trainer import BaseTrainer
from trainer.cmr_wrapper import CMR
from utils import utils, losses, wandb_utils


class FGEL(CMR):
    def __init__(self, data_cfg, model_cfg, exp_cfg) -> None:
        self.estimator_class = NeuralFGEL
        super().__init__(data_cfg, model_cfg, exp_cfg)


class FGELTrainerBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, data_cfg, model_cfg, exp_cfg, **_ignored):
        if not self._instance:
            self._instance = FGEL(data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg)
        return self._instance


if __name__=='__main__':
    import cmr
    print('hallo', cmr)
