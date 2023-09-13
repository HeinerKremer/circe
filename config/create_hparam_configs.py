import argparse

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

def generate_yamls(baseconfig, hparams, cluster_spec=None, seeds=[42]):
    path, file = os.path.split(baseconfig)
    wd = get_wd()
    method = os.path.splitext(file)[0]
    hparam_directory = path + "/" + method +'_hparam'
    absolute_path = os.path.split(os.path.realpath(__file__))[0] + "/" + hparam_directory
    os.makedirs(absolute_path, exist_ok=True)
    os.makedirs(absolute_path + "/cluster", exist_ok=True)
    paths = []

    cfg = yaml.load(open(baseconfig, "r"), Loader=yaml.FullLoader)

    for hparam in iterate_argument_combinations(hparams):
        config = copy.deepcopy(cfg)
        if method in ["gcm", "circe", "hscic"]:
            config['model'].update(hparam)
        else:
            config['model']['trainer_config'].update(hparam)
        config_name = [f'{key}={value}' for key, value in hparam.items()]
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
        # !/bin/bash
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
    args = parser.parse_args()

    if args.method == "vmm":
        baseconfig = "dsprites_linear/vmm.yml"
        hparams = {"reg_param": [1e0],
                    "theta_reg_param": [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                    "progress_bar": [False]}
    elif args.method == "circe":
        baseconfig = "dsprites_linear/circe.yml"
        hparams = {"lamda": [0, 1, 10, 100, 1000]}
    elif args.method == 'hscic':
        baseconfig = "dsprites_linear/hscic.yml"
        hparams = {"lamda": [0, 10, 100, 1000]}
    elif args.method == 'gcm':
        baseconfig = "dsprites_linear/gcm.yml"
        hparams = {"lamda": [0, 1e-2, 1e-4]}
    else:
        raise NotImplementedError('Invalid method specified')

    cluster_spec = {"cpus": 16,
                    "memory": 64000,
                    "gpus": 1,
                    "cuda_memory": 16000}

    seeds = list(range(42, 52))
    print(generate_yamls(baseconfig, hparams, cluster_spec, seeds))


