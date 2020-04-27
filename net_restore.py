# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:21:27 2020

@author: Leon
"""

import numpy as np

def name_mapping(var_dict, debug=False):
    keys = var_dict.keys()
    print(len(keys))
    mapped_dict = {}
    for k in keys:
        key = k.split(':0')[0]
        new_key = key
        print(key)
        if '/W' in key:
            new_key = key.replace('/W', '/weights')
        elif '/mean/EMA' in key:
            new_key = key.replace('/mean/EMA', '/moving_mean')
        elif '/variance/EMA' in key:
            new_key = key.replace('/variance/EMA', '/moving_variance')
        if 'res' in new_key:
            new_key = new_key.replace('res', 'group')
        if '.' in new_key:
            new_key = new_key.replace('.', '/block')
        if 'bnlast' in new_key:
            new_key = new_key.replace('bnlast', 'block1/bnlast')
        if 'group1/bn' in new_key:
            new_key = new_key.replace('block1/bn', 'block1/prebn')        
        mapped_dict[new_key] = var_dict[k]
    if debug:
        mapped_dict['fc/biases'] = var_dict['linear/b:0']
        mapped_dict['fc/weights'] = var_dict['linear/W:0']
    return mapped_dict

net = np.load('./pretrained_models/ResNet18-Preact-Mixup.npz')
#net = np.load('./pretrained_models/ImageNet-ResNet18-Preact.npz')
dict_restore1 = {}
dict_keys = net.keys()
for k in dict_keys:
    dict_restore1[k] = net[k]
net_mapped = name_mapping(dict_restore1, debug=False)
