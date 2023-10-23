import argparse
from collections import defaultdict

import yaml
import os
import copy


def iterate_argument_combinations(argument_dict):
    args = list(argument_dict.values())
    pools = [tuple(pool) for pool in args]
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield {key: val for key, val in zip(list(argument_dict.keys()), prod)}


def get_wd():
    path = os.path.realpath(__file__)
    path, file = os.path.split(path)
    wd, file = os.path.split(path)
    return wd


def adapt_cosine_scheduler_to_epochs(config):
    try:
        config['model']['trainer_config']['theta_optim_args']['scheduler']['CosineAnnealingLR']['T_max'] = config['model']['epochs']
    except KeyError:
        pass

    try:
        config['model']['trainer_config']['dual_optim_args']['scheduler']['CosineAnnealingLR']['T_max'] = config['model']['epochs']
    except KeyError:
        pass

    try:
        config['model']['scheduler']['CosineAnnealingLR']['T_max'] = config['model']['epochs']
    except KeyError:
        pass
    return config


def update_weight_decay(config, weight_decay):
    try:
        config['model']['trainer_config']['theta_optim_args']['optimizer']['AdamW']['weight_decay'] = weight_decay
    except KeyError:
        pass

    try:
        config['model']['trainer_config']['dual_optim_args']['optimizer']['AdamW']['weight_decay'] = weight_decay
    except KeyError:
        pass

    try:
        config['model']['optimizer']['AdamW']['weight_decay'] = weight_decay
    except KeyError:
        pass
    return config


def update_learning_rates(config, lr: tuple):
    try:
        for k in config['model']['trainer_config']['theta_optim_args']['optimizer'].keys():
            config['model']['trainer_config']['theta_optim_args']['optimizer'][k]['lr'] = lr[0]
    except KeyError:
        pass

    try:
        for k in config['model']['trainer_config']['dual_optim_args']['optimizer'].keys():
            config['model']['trainer_config']['dual_optim_args']['optimizer'][k]['lr'] = lr[1]
    except KeyError:
        pass

    try:
        for k in config['model']['optimizer'].keys():
            config['model']['optimizer'][k]['lr'] = lr
    except KeyError:
        pass
    return config


def generate_yamls(exp, method, hparams, cluster_spec=None, seeds=[42]):
    baseconfig = f'{exp}/{method}.yml'
    path, file = os.path.split(baseconfig)
    wd = get_wd()
    method = os.path.splitext(file)[0]
    hparam_directory = path + "/" + method +'_hparam'
    absolute_path = os.path.split(os.path.realpath(__file__))[0] + "/" + hparam_directory
    os.makedirs(absolute_path, exist_ok=True)
    os.makedirs(absolute_path + "/cluster", exist_ok=True)
    paths = []

    cfg = yaml.load(open(baseconfig, "r"), Loader=yaml.FullLoader)

    if 'learning_rates' not in hparams:
        hparams['learning_rates'] = [None]
    if 'weight_decay' not in hparams:
        hparams['weight_decay'] = [None]
    if 'trainer_config' not in hparams:
        hparams['trainer_config'] = {'': ['']}
    if 'config' not in hparams:
        hparams['config'] = {'': ['']}

    for hparam in iterate_argument_combinations(hparams['config']):
        for hparam_trainer in iterate_argument_combinations(hparams['trainer_config']):
            version = 0
            for lr in hparams['learning_rates']:
                for weight_decay in hparams['weight_decay']:
                    version += 1
                    config = copy.deepcopy(cfg)
                    config['model'].update(hparam)
                    try:
                        config['model']['trainer_config'].update(hparam_trainer)
                    except KeyError:
                        pass
                    config = adapt_cosine_scheduler_to_epochs(config)

                    if lr is not None:
                        config = update_learning_rates(config, lr)
                    if weight_decay is not None:
                        config = update_weight_decay(config, weight_decay)

                    config_name = ([f'{key}={value}' for key, value in hparam.items()]
                                   + [f'{key}={value}' for key, value in hparam_trainer.items()]
                                   + [f'v{version}'])
                    filename = method + "_" + '_'.join(config_name) + ".yml"
                    with open(hparam_directory + "/" + filename, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                        paths.append(hparam_directory + "/" + filename)

    if cluster_spec:
        with open(hparam_directory + "/" + method + '.sub', 'w') as subfile:
            subfile.write(f'executable = {wd}/py3venv.sh\n'
                          + f'arguments = "main.py -v $(config) --seed $(seed) -w"\n'
                          + f'error = {absolute_path}' + '/cluster/$(Name).err\n'
                          + f'output = {absolute_path}' + '/cluster/$(Name).out\n'
                          + f'log = {absolute_path}' + '/cluster/$(Name).log\n'
                          + f'request_cpus = {cluster_spec["cpus"]}\n'
                          + f'request_memory = {cluster_spec["memory"]}\n'
                          + f'request_gpus = {cluster_spec["gpus"]}\n'
                          + f'requirements = TARGET.CUDAGlobalMemoryMb > {cluster_spec["cuda_memory"]}\n')
            for config_path in paths:
                for seed in seeds:
                    name = os.path.splitext(os.path.split(os.path.realpath(config_path))[1])[0]
                    print(name)
                    subfile.write(f'\nconfig = {config_path}\nName = {name}\nseed = {seed}\nqueue\n')

        # Write bashscript to run python in venv
        if not os.path.exists(f'{wd}/py3venv.sh'):
            with open(f'{wd}/py3venv.sh', 'w') as exe:
                exe.write("#!/bin/bash\n"
                          + f"cd {wd}\n"
                          + f"source {wd}/circe_venv/bin/activate\n"
                          + f'echo "python3 $@"\n'
                          + f'python3 $@')
            st = os.stat(f'{wd}/py3venv.sh')
            os.chmod(f'{wd}/py3venv.sh', st.st_mode | 0o111)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--rollouts', type=int, default=10)
    args = parser.parse_args()

    if 'vmm' in args.method:
        # hparams = {
        #     'config': {'epochs': [200, 500]},
        #     'trainer_config': {
        #         "reg_param": [1e-6, 1e-4, 1e-2, 1],
        #         "theta_reg_param": [1e-6, 1e-3, 1],
        #         "progress_bar": [False]},
        #     'learning_rates': [(1e-4, 1e-4), (1e-3, 1e-3), (1e-2, 1e-2), (1e-4, 1e-3), (1e-3, 1e-2)],
        #     'weight_decay': [1e-2, 1e-4]
        # }
        if args.exp == 'dsprites_tricky':
            hparams = {
                'config': {'epochs': [100]},
                'trainer_config': {
                    "progress_bar": [False]},
            }
        else:
            hparams = {
                'config': {},   # {'epochs': [100, 300]},
                'trainer_config': {
                    "reg_param": [1, 10, 50, 100],
                    "progress_bar": [False]},
                # 'learning_rates': [(1e-3, 1e-3), (1e-3, 1e-2)],  #[(1e-4, 1e-4), (1e-3, 1e-3), (1e-4, 1e-3)],
                # 'weight_decay': [1e-2, 1e-4]
            }
            hparams = {}
    elif 'fgel' in args.method:
        hparams = {
            'config': {'epochs': [100],
                       'patience': [100]},
            'trainer_config': {
                "reg_param": [1, 10, 100],
                "divergence": ['chi2', 'kl', 'log'],
                # "theta_reg_param": [1e-6, 1e-3, 1],
                "progress_bar": [False]},
            'learning_rates': [(1e-5, 1e-5), (1e-5, 1e-4)],
        }
    elif args.method == "circe":
        # hparams = {
        #     'config': {
        #         'epochs': [200, 500],
        #         "lamda": [0, 1, 10, 100, 1000],
        #         "progress_bar": [False]
        #     },
        #     'learning_rates': [1e-4, 1e-3, 1e-2],
        #     'weight_decay': [1e-2, 1e-4],
        # }
        if args.exp == 'dsprites_tricky':
            hparams = {
            'config': {
                'epochs': [100],
                "lamda": [0, 1000],
                "progress_bar": [False]
                },
            }
        else:
            hparams = {
                'config': {
                    'epochs': [100, 300],
                    "lamda": [0, 1000],
                    "progress_bar": [False]
                },
                'learning_rates': [1e-4, 1e-3, 1e-2],
                'weight_decay': [1e-2, 1e-4],
            }

    elif args.method == 'hscic':
        hparams = {'config': {
            'epochs': [100],
            "lamda": [1000],
            "progress_bar": [False]},
        }
    elif args.method == 'gcm':
        hparams = {'config': {
            "lamda": [0, 1e-2, 1e-4],
            "progress_bar": [False]},
        }
    elif args.method == 'smm':
        hparams = {
            'config': {'epochs': [200, 500]},
            'trainer_config': {
                "reg_param": [1e-4, 1e-2, 1e0],
                "theta_reg_param": [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                "continuous_updating": [True, False],
                "progress_bar": [False]}
        }
    else:
        raise NotImplementedError('Invalid method specified')

    cluster_spec = {"cpus": 16,
                    "memory": 64000,
                    "gpus": 1,
                    "cuda_memory": 16000}

    seeds = list(range(42, 42 + args.rollouts))
    yamls = generate_yamls(args.exp, args.method, hparams, cluster_spec, seeds)
    print(yamls)
    print(f'Generated {len(yamls)} yamls')


