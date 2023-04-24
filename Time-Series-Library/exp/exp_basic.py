import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
    
from data_provider.data_factory import data_provider

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.data_set, self.data_loader = data_provider(self.args, 'train')
        self.train_valid_split_ratio = 0.15
        self.train_valid_dataset, self.test_dataset = train_test_split(self.data_set, test_size=self.train_valid_split_ratio, shuffle=False)  # False
        self.train_dataset, self.valid_dataset = train_test_split(self.train_valid_dataset, test_size=self.train_valid_split_ratio,shuffle=False) # False
        # Data Loader
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size,shuffle=True,num_workers=self.args.num_workers,drop_last=True)
        self.valid_loader = DataLoader(dataset=self.valid_dataset, batch_size=self.args.batch_size,shuffle=True,num_workers=self.args.num_workers,drop_last=False)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.args.batch_size,shuffle=True,num_workers=self.args.num_workers,drop_last=False)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
