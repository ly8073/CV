#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/24 23:47
# @Author  : Ly
# @File    : ModelGenerater.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
from config.ResNetConfigs import resnet_dict


class ModelGenerater:
    model_gene_dict = {
        'ResNet': resnet_dict
    }

    @classmethod
    def get_model(cls, model_name, num_class):
        net_name, net_dep = model_name.split('_')
        model_config = cls.model_gene_dict[net_name][net_dep]()
        net = cls.model_gene_dict[net_name]['gene'](model_config, num_class)
        return net
