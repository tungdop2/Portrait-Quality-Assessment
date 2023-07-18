import argparse
import os
import copy
from omegaconf import OmegaConf

import torch
from torch import nn
from torchvision import transforms

from utils import instantiate_from_config, merge_configs

BASE_CONFIG_PATH = "configs/base.yaml"

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--config", type=str, default='configs/resnet/resnet50.yaml')

    return parser

def instantiate_model_from_config(config):
    backbone_config = config.pop('backbone')
    backbone = instantiate_from_config(backbone_config)
    
    model = instantiate_from_config(config, backbone=backbone)
 
    return model

def main(args):
    config = OmegaConf.load(args.config)
    base_config = OmegaConf.load(BASE_CONFIG_PATH)
    config = merge_configs(base_config, config)
    raw_config = copy.deepcopy(config)
    model_config = config.pop('model')

    model = instantiate_model_from_config(model_config)

    data_config = config.pop('data')

    data_module = instantiate_from_config(data_config)
    
    loggers = []
    loggers_config = config.pop('loggers')
    for logger_name in loggers_config:
        logger = instantiate_from_config(loggers_config[logger_name])
        loggers.append(logger)
        if logger_name == 'tensorboard':
            os.makedirs(logger.log_dir, exist_ok=True)
            OmegaConf.save(raw_config, logger.log_dir + '/config.yaml')
        
    callbacks = []
    callbacks_config = config.pop('callbacks')
    for callback_name in callbacks_config:
        callback = instantiate_from_config(callbacks_config[callback_name])
        callbacks.append(callback)

    trainer_config = config.pop('trainer')
    trainer = instantiate_from_config(trainer_config, logger=loggers, callbacks=callbacks)

    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)