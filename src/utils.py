import logging
import logzero
import json
import subprocess
import sys
import time
from collections import OrderedDict, deque
from pathlib import Path
from sklearn.model_selection import ParameterGrid, ParameterSampler
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import numpy as np
import pandas as pd


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


def accuracy(output, target, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    if len(res) == 1:
        res = res[0]

    return res


def save_checkpoint(model, epoch, filename, optimizer=None):
    if optimizer is None:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, filename)
    else:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename)


def load_checkpoint(model, path, optimizer=None):
    resume = torch.load(path)

    if ('module' in list(resume['state_dict'].keys())[0]) \
            and not (isinstance(model, torch.nn.DataParallel)):
        new_state_dict = OrderedDict()
        for k, v in resume['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(resume['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(resume['optimizer'])
        return model, optimizer
    else:
        return model


def set_logger(log_dir, loglevel=logging.INFO, tf_board_dir=None):
    if not Path(log_dir).exists():
        Path(log_dir).mkdir(parents=True)
    logzero.loglevel(loglevel)
    logzero.formatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s'))
    logzero.logfile(log_dir + '/logfile')

    if tf_board_dir is not None:
        if not Path(tf_board_dir).exists():
            Path(tf_board_dir).mkdir(parents=True)
        writer = SummaryWriter(tf_board_dir)

        return writer


def get_optim(params, model):
    if params['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), params['lr'], weight_decay=params['wd'])
    elif params['optimizer'] == 'momentum':
        optimizer = optim.SGD(model.parameters(), params['lr'], momentum=0.9, weight_decay=params['wd'])
    elif params['optimizer'] == 'nesterov':
        optimizer = optim.SGD(model.parameters(), params['lr'], momentum=0.9,
                              weight_decay=params['wd'], nesterov=True)
    elif params['optimizer'] == 'adam':
        optimizer = optim.Adam(model.params, params['lr'], weight_decay=params['wd'])
    elif params['optimizer'] == 'amsgrad':
        optimizer = optim.Adam(model.params, params['lr'], weight_decay=params['wd'], amsgrad=True)
    elif params['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters, params['lr'], weight_decay=params['wd'])
    else:
        raise ValueError

    return optimizer


# check if current params combination has already done
def check_duplicate(df, p, space):
    # new key is included
    if not all(map(lambda x: x in df.columns, space.keys())):
        return False
    for i in range(len(df)):  # for avoiding unexpected cast due to row-slicing
        is_dup = True
        for key, val in p.items():
            if df.loc[i, key] != val:
                is_dup = False
                break
        if is_dup:
            return True
    return False


def launch_tuning(mode, n_iter, n_gpu, devices, params, space, root, metrics=('acc', 'loss')):
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
        for m in metrics:
            cols.append(m)
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
