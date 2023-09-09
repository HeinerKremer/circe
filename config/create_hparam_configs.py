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

def generate_yamls(baseconfig, hparams, cluster_spec=None):
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
        config['model']['trainer_config'].update(hparam)
        config_name = [f'{key}={value}' for key, value in hparam.items()]
        filename = method + "_" + '_'.join(config_name) + ".yml"
        with open(hparam_directory + "/" + filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            paths.append(hparam_directory + "/" + filename)

    if cluster_spec:
        with open(hparam_directory + "/" + method + '.sub', 'w') as subfile:
            subfile.write(f'executable = {wd}/py3venv.sh\n'
                          + f'arguments = "main.py -v $(MyArg) --seed 42 -w"\n'
                          + f'error = {absolute_path}' + '/cluster/$(MyArg).err\n'
                          + f'output = {absolute_path}' + '/cluster/$(MyArg).out\n'
                          + f'log = {absolute_path}' + '/cluster/$(MyArg).log\n'
                          + f'request_cpus = {cluster_spec["cpus"]}\n'
                          + f'request_memory = {cluster_spec["memory"]}\n'
                          + f'request_gpus = {cluster_spec["gpus"]}\n'
                          + f'requirements = TARGET.CUDAGlobalMemoryMb > {cluster_spec["cuda_memory"]}\n')
            for path in paths:
                subfile.write(f'\nMyArg = {path} \nqueue\n')

        # Write bashscript to run python in venv
        # !/bin/bash
        with open(f'{wd}/py3venv.sh', 'w') as exe:
            exe.write("#!/bin/bash\n"
                      + f"cd {wd}\n"
                      + f"source {wd}/circe_venv/bin/activate\n"
                      + f'echo "python3 $@"\n'
                      + f'python3 $@')
    return paths


if __name__ == "__main__":
    # Hparam sweeps
    hparams = {'reg_param': [1e-6, 1e-4, 1e-2, 1],
               'theta_reg_param': [1e-4, 1e-2, 1, 1e2, 1e4]}

    baseconfig = "dsprites_linear/vmm.yml"
    cluster_spec = {"cpus": 16,
                    "memory": 64000,
                    "gpus": 1,
                    "cuda_memory": 16000}

    print(generate_yamls(baseconfig, hparams, cluster_spec))


