import json
import logging
import subprocess
import sys
import time
from collections import OrderedDict, deque
from pathlib import Path

import logzero
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import ParameterGrid, ParameterSampler
from tensorboardX import SummaryWriter


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model, epoch, filename, optimizer=None, save_arch=False, params=None):
    attributes = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }

    if optimizer is not None:
        attributes['optimizer'] = optimizer.state_dict()

    if save_arch:
        attributes['arch'] = model

    if params is not None:
        attributes['params'] = params

    try:
        torch.save(attributes, filename)
    except TypeError:
        if 'arch' in attributes:
            print('Model architecture will be ignored because the architecture includes non-pickable objects.')
            del attributes['arch']
            torch.save(attributes, filename)


def load_checkpoint(path, model=None, optimizer=None, params=False):
    resume = torch.load(path)

    rets = dict()

    if model is not None:
        if ('module' in list(resume['state_dict'].keys())[0]) \
                and not (isinstance(model, torch.nn.DataParallel)):
            new_state_dict = OrderedDict()
            for k, v in resume['state_dict'].items():
                new_state_dict[k.replace('module.', '')] = v  # remove DataParallel wrapping

            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(resume['state_dict'])

        rets['model'] = model

    if optimizer is not None:
        optimizer.load_state_dict(resume['optimizer'])
        rets['optimizer'] = optimizer
    if params:
        rets['params'] = resume['params']

    return rets


def load_model(path, is_inference=True):
    resume = torch.load(path)
    model = resume['arch']
    model.load_state_dict(resume['state_dict'])
    if is_inference:
        model.eval()
    return model


def get_logger(log_dir, loglevel=logging.INFO, tensorboard_dir=None):
    from logzero import logger

    if not Path(log_dir).exists():
        Path(log_dir).mkdir(parents=True)
    logzero.loglevel(loglevel)
    logzero.logfile(log_dir + '/logfile')

    if tensorboard_dir is not None:
        if not Path(tensorboard_dir).exists():
            Path(tensorboard_dir).mkdir(parents=True)
        writer = SummaryWriter(tensorboard_dir)

        return logger, writer

    return logger


def get_optim(params, target):

    assert isinstance(target, nn.Module) or isinstance(target, dict)

    if isinstance(target, nn.Module):
        target = target.parameters()

    if params['optimizer'] == 'sgd':
        optimizer = optim.SGD(target, params['lr'], weight_decay=params['wd'])
    elif params['optimizer'] == 'momentum':
        optimizer = optim.SGD(target, params['lr'], momentum=0.9, weight_decay=params['wd'])
    elif params['optimizer'] == 'nesterov':
        optimizer = optim.SGD(target, params['lr'], momentum=0.9,
                              weight_decay=params['wd'], nesterov=True)
    elif params['optimizer'] == 'adam':
        optimizer = optim.Adam(target, params['lr'], weight_decay=params['wd'])
    elif params['optimizer'] == 'amsgrad':
        optimizer = optim.Adam(target, params['lr'], weight_decay=params['wd'], amsgrad=True)
    elif params['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(target, params['lr'], weight_decay=params['wd'])
    else:
        raise ValueError

    return optimizer


def write_tuning_result(params: dict, result: dict, df_path: str):
    row = pd.DataFrame()
    for key in params['tuning_params']:
        row[key] = [params[key]]

    for key, val in result.items():
        row[key] = val

    with lockfile.FileLock(df_path):
        df_results = pd.read_csv(df_path)
        df_results = pd.concat([df_results, row], sort=False).reset_index(drop=True)
        df_results.to_csv(df_path, index=None)


def check_duplicate(df: pd.DataFrame, p: dict, space):
    """check if current params combination has already done"""

    new_key_is_included = not all(map(lambda x: x in df.columns, space.keys()))
    if new_key_is_included:
        return False

    for i in range(len(df)):  # for avoiding unexpected cast due to row-slicing
        is_dup = True
        for key, val in p.items():
            if df.loc[i, key] != val:
                is_dup = False
                break
        if is_dup:
            return True
    else:
        return False


def launch_tuning(mode: str, n_iter: int, n_gpu: int, devices: str,
                  params: dict, space: dict, root):
    """
    Launch paramter search by specific way.
    Each trials are launched asynchronously by forking subprocess and all results of trials
    are automatically written in csv file.

    :param mode: the way of parameter search, one of 'grid or random'.
    :param n_iter: num of iteration for random search.
    :param n_gpu: num of gpu used at one trial.
    :param devices: gpu devices for tuning.
    :param params: training parameters.
                   the values designated as tuning parameters are overwritten
    :param space: paramter search space.
    :param root: path of the root directory.
    """

    gpu_list = deque(devices.split(','))

    if mode == 'grid':
        param_list = list(ParameterGrid(space))
    elif mode == 'random':
        param_list = list(ParameterSampler(space, n_iter))
    else:
        raise ValueError

    params['tuning_params'] = list(param_list[0].keys())

    df_path = root+f'experiments/{params["ex_name"]}/tuning/results.csv'
    if Path(df_path).exists() and Path(df_path).stat().st_size > 5:
        df_results = pd.read_csv(df_path)
    else:
        cols = list(param_list[0].keys())
        df_results = pd.DataFrame(columns=cols)
        df_results.to_csv(df_path, index=False)

    procs = []
    for p in param_list:

        if check_duplicate(df_results, p, param_list[0]):
            print(f'skip: {p} because this setting is already experimented.')
            continue

        # overwrite hyper parameters for search
        for key, val in p.items():
            params[key] = val

        while True:
            if len(gpu_list) >= n_gpu:
                devices = ','.join([gpu_list.pop() for _ in range(n_gpu)])
                params_path = root + f'experiments/{params["ex_name"]}/tuning/params_{devices[0]}.json'
                with open(params_path, 'w') as f:
                    json.dump(params, f)
                break
            else:
                time.sleep(1)
                for i, (p, dev) in enumerate(procs):
                    if p.poll() is not None:
                        gpu_list += deque(dev.split(','))
                        del procs[i]

        cmd = f'{sys.executable} {params["ex_name"]}.py job ' \
              f'--tuning --params-path {params_path} --devices "{devices}"'
        procs.append((subprocess.Popen(cmd, shell=True), devices))

    while True:
        time.sleep(1)
        if all(p.poll() is not None for i, (p, dev) in enumerate(procs)):
            print('All parameter combinations have finished.')
            break
    
    show_tuning_result(params["ex_name"])


def show_tuning_result(ex_name, mode='markdown', sort_by=None, ascending=False):
    
    table = pd.read_csv(f'../experiments/{ex_name}/tuning/results.csv')
    if sort_by is not None:
        table = table.sort_values(sort_by, ascending=ascending)
    
    if mode == 'markdown':
        from tabulate import tabulate
        print(tabulate(table, headers='keys', tablefmt='pipe', showindex=False))
    elif mode == 'latex':
        from tabulate import tabulate
        print(tabulate(table, headers='keys', tablefmt='latex', floatfmt='.2f', showindex=False))
    else:
        from IPython.core.display import display
        display(table)
