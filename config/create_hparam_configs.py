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


def generate_yamls(baseconfig, hparams, cluster_spec=None):
    path, file = os.path.split(baseconfig)
    method = os.path.splitext(file)[0]
    hparam_directory = path + "/" + method +'_hparam'
    absolute_path = os.path.split(os.path.realpath(__file__))[0] + "/" + hparam_directory
    os.makedirs(absolute_path, exist_ok=True)
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
            subfile.write(f'executable = py3venv.sh\n'
                          + f'arguments = "main.py -v $(config_file) --seed 42 -w"\n'
                          # + f'error = {path}/cluster/jobs_{experiment_name}/{filename}.err\n'
                          # + f'output = {path}/cluster/jobs_{experiment_name}/{filename}.out\n'
                          # + f'log = {path}/cluster/jobs_{experiment_name}/{filename}.log\n'
                          + f'request_cpus = {cluster_spec["cpus"]}\n'
                          + f'request_memory = {cluster_spec["memory"]}\n'
                          + f'request_gpus = {cluster_spec["gpus"]}\n'
                          + f'requirements = TARGET.CUDAGlobalMemoryMb > {cluster_spec["cuda_memory"]}\n'
                          + "max_materialize = 150\n")
            for path in paths:
                subfile.write(f'\n\nconfig_file={path} \nqueue')
    return paths


if __name__ == "__main__":
    # Hparam sweeps
    hparams = {'reg_param': [1e-6, 1e-4, 1e-2, 1],
               'theta_reg_param': [1e-4, 1e-2, 1, 1e2, 1e4]}

    baseconfig = "dsprites_linear/vmm.yml"
    cluster_spec = {"cpus": 16,
                    "memory": 64000,
                    "gpus": 1,
                    "cuda_memory": 20000}

    print(generate_yamls(baseconfig, hparams, cluster_spec))


