import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_results(exp, methods):
    results = {}

    for method in methods:
        path = f'{exp}/{method}_'
        results[method] = {'test': [], 'test_ood': []}
        for mode in ['test', 'test_ood']:
            file = path + mode + '.csv'
            with open(file, newline='') as csvfile:
                file = csv.reader(csvfile, delimiter=' ')
                for i, row in enumerate(file):
                    if i == 0:
                        continue
                    results[method][mode].append(float(row[0].split(sep=",")[2].replace('"', '')))
    return results

    #     results_agg = {}
    #     for mode, res in results.items():
    #         results_agg[mode] = {}
    #         for r, v in res.items():
    #             results_agg[mode][r] = {'mean': np.mean(v),
    #                                     'std': np.std(v),
    #                                     'sem': np.std(v)/np.sqrt(len(v)),
    #                                     'max': np.max(v),
    #                                     'min': np.min(v),
    #                                     'median': np.median(v),
    #                                     'runs': v}
    #     results_plotting = {}
    #     for mode, res in results_agg.items():
    #         results_plotting[mode] = defaultdict(list)
    #         for r, v in res.items():
    #             results_plotting[mode]['reg_param'].append(r)
    #             results_plotting[mode]['mean'].append(v['mean'])
    #             results_plotting[mode]['std'].append(v['std'])
    #             results_plotting[mode]['sem'].append(v['sem'])
    #             results_plotting[mode]['max'].append(v['max'])
    #             results_plotting[mode]['min'].append(v['min'])
    #             results_plotting[mode]['median'].append(v['median'])
    #
    #         sorted_indices = np.argsort(results_plotting[mode]['reg_param'])
    #         for k in results_plotting[mode].keys():
    #             results_plotting[mode][k] = np.asarray(results_plotting[mode][k])[sorted_indices]
    #
    # return results_agg, results_plotting


def plot_results(results, prop='mean', title=None):
    sns.set_theme()
    fig, ax = plt.subplots(1, 1)

    for mode, res in results.items():
        ax.plot(res['reg_param'], res[prop], label=mode)
        ax.fill_between(res['reg_param'],
                        np.subtract(res['mean'], res['std']),
                        np.add(res['mean'], res['std']),
                        alpha=0.2)

    ax.set_xscale('log')
    ax.legend()
    # ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim([0, 0.155])
    ax.yaxis.set_ticks(np.arange(0, 0.175, 0.025))
    if title:
        plt.title(title)
    plt.savefig(title + '.pdf', dpi=200)
    plt.show()


if __name__ == "__main__":
    exp = 'dsprites_tricky'
    methods = ['circe', 'vmm']

    results = load_results(exp, methods)
    print(results)
    # plot_results(results_plotting, title=f'{exp}: {method}')