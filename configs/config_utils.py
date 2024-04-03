import os
import yaml
import logging
from datetime import datetime

def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

class CONFIG(object):
    '''
    Stores all configures
    '''
    def __init__(self, input=None):
        '''
        Loads config file
        :param path (str): path to config file
        :return:
        '''
        self.config = self.read_to_dict(input)

    def read_to_dict(self, input):
        if not input:
            return dict()
        if isinstance(input, str) and os.path.isfile(input):
            if input.endswith('yaml'):
                with open(input, 'r') as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
            else:
                ValueError('Config file should be with the format of *.yaml')
        elif isinstance(input, dict):
            config = input
        else:
            raise ValueError('Unrecognized input type (i.e. not *.yaml file nor dict).')

        return config

    def update_config(self, *args, **kwargs):
        '''
        update config and corresponding logger setting
        :param input: dict settings add to config file
        :return:
        '''
        cfg1 = dict()
        for item in args:
            cfg1.update(self.read_to_dict(item))

        cfg2 = self.read_to_dict(kwargs)

        new_cfg = {**cfg1, **cfg2}

        update_recursive(self.config, new_cfg)
        # when update config file, the corresponding logger should also be updated.
        self.__update_logger()

    def write_config(self,save_path):
        with open(save_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style = False)