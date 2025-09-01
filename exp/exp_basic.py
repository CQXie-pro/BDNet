import os
import torch
from models import Autoformer, DLinear, FEDformer, Informer, PatchTST, MICN, iTransformer, FreTS, FITS, RLinear, Linear, NLinear, BDNet, SparseTSF, CycleNet, Leddam


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'PatchTST': PatchTST,
            'MICN': MICN,
            'iTransformer': iTransformer,
            'FreTS': FreTS,
            'FITS': FITS,
            'RLinear': RLinear,
            'Linear': Linear,
            'NLinear': NLinear,
            'BDNet': BDNet,
            'SparseTSF': SparseTSF,
            'CycleNet': CycleNet,
            'Leddam': Leddam
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

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
