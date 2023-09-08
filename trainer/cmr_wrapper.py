import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
from scipy.linalg import solve as scp_solve
from cmr.methods.vmm_neural import NeuralVMM

from trainer.base_trainer import BaseTrainer
from utils import utils, losses, wandb_utils


class CMR(BaseTrainer):
    def __init__(self, data_cfg, model_cfg, exp_cfg) -> None:
        super().__init__(data_cfg, model_cfg, exp_cfg)
        self.estimator = self._construct_estimator()

    @property
    def opt(self):
        return self.estimator.theta_optimizer

    def _construct_estimator(self):
        regularizer = None
        if self.model_cfg.model_key == 'regressor':
            def moment_function(y_pred, y_true):
                return y_pred[1] - y_true

            if self.model_cfg.trainer_config["erm_reg_param"] > 0:
                def regularizer(y_pred, y_true):
                    return torch.nn.functional.mse_loss(y_pred[1], y_true)
        elif self.model_cfg.model_key == 'classifier':
            raise NotImplementedError(
                'Need to think how to implement CMR classification. E.g. psi(x,y)=(p_1,p_2) - (1, 0).')
        else:
            raise NotImplementedError('Moment function not specified.')

        print('Model config: ', self.model_cfg.trainer_config)
        estimator = NeuralVMM(model=self.model, moment_function=moment_function, theta_regularizer=regularizer,
                              theta_reg_param=self.model_cfg.trainer_config["erm_reg_param"],
                              **self.model_cfg.trainer_config)
        return estimator

    def _set_kernels(self):
        pass

    def _get_yz_regressors(self):
        pass

    def _scheduler_step(self):
        self.estimator._scheduler_step()

    def _epoch(self, epochID, mode):
        '''
        Run a single epoch, aggregate losses & log to wandb.
        '''
        train = 'train' in mode
        self.model.train() if train else self.model.eval()

        all_losses = defaultdict(list)

        data_iter = iter(self.dataloaders[mode])
        tqdm_iter = tqdm(range(len(self.dataloaders[mode])), dynamic_ncols=True)

        for i in tqdm_iter:
            batch = utils.dict_to_device(next(data_iter), self.device)
            x, y, z = batch['x'], batch['y'], batch['z']

            # For CMR estimators the optimization objective differs from the target objective
            with torch.no_grad():
                _, y_ = self.model(x)
                if self.model_cfg.model_key == 'regressor':
                    target_loss = F.mse_loss(y_, y)
                    moment_norm = self.estimator._calc_val_moment_violation([x, y], z)
                    # mmr = self.estimator._calc_val_mmr([x, y], z)
                    # hsic = self.estimator._calc_val_hsic([x, y], z)
                elif self.model_cfg.model_key == 'classifier':
                    raise NotImplementedError('Classification not yet implemented.')
                    # label = batch['label']
                    # label[label > self.model_cfg.target_threshold] = 1
                    # target_loss = F.nll_loss(y_, label)

            if train:
                cmr_obj = self.estimator._optimize_step_theta([x, y], z)
            else:
                with torch.no_grad():
                    obj_theta, _ = self.estimator.objective([x, y], z)
                    cmr_obj = float(obj_theta.detach().cpu().numpy())

            tqdm_iter.set_description("V: {} | Epoch: {} | {} | Obj: {:.4f} | Target Loss: {:.4f}".format(
                self.exp_cfg.version, epochID, mode, cmr_obj, target_loss.item()
            ), refresh=True)

            all_losses['target_loss'].append(target_loss.item())
            all_losses['moment_norm'].append(moment_norm)
            # all_losses['mmr'].append(mmr)
            # all_losses['hsic'].append(hsic)
            all_losses['cmr_obj'].append(cmr_obj)

        all_losses = utils.aggregate(all_losses)
        if self.exp_cfg.wandb:
            wandb_utils.log_epoch_summary(epochID, mode, all_losses)

        return all_losses['target_loss']


class CMRTrainerBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, data_cfg, model_cfg, exp_cfg, **_ignored):
        if not self._instance:
            self._instance = CMR(data_cfg=data_cfg, model_cfg=model_cfg, exp_cfg=exp_cfg)
        return self._instance


if __name__=='__main__':
    import cmr
    print('hallo', cmr)
