import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
from scipy.linalg import solve as scp_solve

from trainer.base_trainer import BaseTrainer
from utils import utils, losses, wandb_utils


class CMR(BaseTrainer):
    def __init__(self, data_cfg, model_cfg, exp_cfg) -> None:
        super().__init__(data_cfg, model_cfg, exp_cfg)
        self.estimator = self._construct_estimator()
        self.tqdm = self.model_cfg.trainer_config["progress_bar"]

    @property
    def opt(self):
        return self.estimator.theta_optimizer

    def _construct_estimator(self):
        if self.model_cfg.model_key == 'regressor':
            def moment_function(y_pred, y_true):
                return y_pred[1] - y_true

            def target_loss(y_pred, y_true):
                return torch.nn.functional.mse_loss(y_pred[1], y_true)
        elif self.model_cfg.model_key == 'classifier':
            raise NotImplementedError(
                'Need to think how to implement CMR classification. E.g. psi(x,y)=(p_1,p_2) - (1, 0).')
        else:
            raise NotImplementedError('Moment function not specified.')

        print('Model config: ', self.model_cfg.trainer_config)
        if not self.estimator_class:
            raise NotImplementedError('Need to specify CMR estimator class.')

        self.model_cfg.trainer_config['wandb'] = self.exp_cfg.wandb
        self.model_cfg.trainer_config['val_loss_func'] = target_loss
        self.model_cfg.trainer_config['batch_size'] = self.data_cfg.batch_size
        self.model_cfg.trainer_config['num_workers'] = self.data_cfg.num_workers

        if self.model_cfg.trainer_config["theta_reg_param"] > 0:
            self.model_cfg.trainer_config['theta_regularizer'] = target_loss
        else:
            self.model_cfg.trainer_config['theta_regularizer'] = None

        estimator = self.estimator_class(model=self.model,
                                         datasets=self.datasets,
                                         moment_function=moment_function,
                                         **self.model_cfg.trainer_config)
        return estimator

    def _set_kernels(self):
        pass

    def _get_yz_regressors(self):
        pass

    def _scheduler_step(self):
        self.estimator._scheduler_step()

    def _epoch(self, epochID, mode):
        return self.estimator._epoch(epochID, mode)

    # def _epoch(self, epochID, mode):
    #     '''
    #     Run a single epoch, aggregate losses & log to wandb.
    #     '''
    #     train = 'train' in mode
    #     self.model.train() if train else self.model.eval()
    #
    #     all_losses = defaultdict(list)
    #
    #     data_iter = iter(self.dataloaders[mode])
    #
    #     if self.tqdm:
    #         tqdm_iter = tqdm(range(len(self.dataloaders[mode])), dynamic_ncols=True)
    #     else:
    #         tqdm_iter = range(len(self.dataloaders[mode]))
    #
    #     for i in tqdm_iter:
    #         batch = utils.dict_to_device(next(data_iter), self.device)
    #         batch = {'t': batch['x'], 'y': batch['y'], 'z': batch['z']}
    #
    #         if train:
    #             cmr_obj = self.estimator._train_step_model(batch)
    #         else:
    #             obj_theta, _ = self.estimator.objective(batch)
    #             val_loss = self.estimator.calc_validation_metric(batch)
    #             cmr_obj = float(obj_theta.detach().cpu().numpy())
    #
    #         if self.tqdm:
    #             tqdm_iter.set_description("V: {} | Epoch: {} | {} | Obj: {:.4f} | Target Loss: {:.4f}".format(
    #                 self.exp_cfg.version, epochID, mode, cmr_obj, val_loss
    #             ), refresh=True)
    #
    #         all_losses['target_loss'].append(val_loss)
    #         all_losses['objective'].append(cmr_obj)
    #
    #     all_losses = utils.aggregate(all_losses)
    #     if self.exp_cfg.wandb:
    #         wandb_utils.log_epoch_summary(epochID, mode, all_losses)
    #
    #     return all_losses['target_loss']


if __name__ == '__main__':
    import cmr
    print('hallo', cmr)
