import importlib
from omegaconf import DictConfig, OmegaConf

def instantiate_from_config(config, **kwargs):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **kwargs)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def merge_configs(base_config, override_config):
    base_config = OmegaConf.create(base_config)
    override_config = OmegaConf.create(override_config)
    return OmegaConf.merge(base_config, override_config)