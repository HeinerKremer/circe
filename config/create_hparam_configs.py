import yaml
import os


# Hparam sweeps
hparams = {'reg_param': [1e-6, 1e-4, 1e-2, 1],
           'dual_optim_args': ['optimizer': {}]}



baseconfig = "dsprites_linear/vmm.yml"
cfg = yaml.load(open(baseconfig, "r"), Loader=yaml.FullLoader)

configs = [
    {},
    {'optimizer': {'OAdam'}}
]


