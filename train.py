import argparse
import collections
import torch
import numpy as np
import GPUtil
import os

import dataset 
import dataloader
from dataloader.subset import random_split
import module.metric as module_metric
import module.model as module_model
import module.loss as module_loss
from utils import setup_seed
from utils.parse_config import ConfigParser
from trainer import Trainer


def main(config):
    logger = config.get_logger('train')

    # selece gpus
    devices = config.init_obj('GPUtil', GPUtil)
    devices_str = ','.join(map(str, devices))
    os.environ['CUDA_VISIBLE_DEVICES'] = devices_str
    logger.info('Use gpus: {}'.format(devices_str))
    
    # fix random seeds for reproducibility
    if config['seed'] is not None:
        setup_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # load and split data into trainset and validset by valid_split
    fullset = config.init_obj('dataset', dataset)
    trainset, ph1 = random_split(fullset, config['dataset']['valid_split']) # placeholder
    validset = config.init_obj('valid_dataset', dataset)
    validset, ph2 = random_split(validset, config['valid_dataset']['valid_split'])
#    fullset = config.init_obj('dataset', dataset)
#    trainset, validset = random_split(fullset, config['dataset']['valid_split'])
    trainloader = config.init_obj('dataloader', dataloader, trainset)
    if validset is None:
        validloader = None
    else:
        validloader = config.init_obj('dataloader', dataloader, validset)

    # build model architecture, then print to console
    model = config.init_obj('model', module_model)
    # logger.info(model)

    # get function handles of loss and metrics
    criterion = config.init_obj('loss', module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      trainloader=trainloader,
                      validloader=validloader,
                      lr_scheduler=lr_scheduler)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-n', '--name'], type=str, target='name'),
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='dataloader;args;batch_size'),
        CustomArgs(['--u2w', '--utt2wav_val'], type=str, target='valid_dataset;args;wav_scp'),
        CustomArgs(['--u2l', '--utt2label_val'], type=str, target='valid_dataset;args;utt2label')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
