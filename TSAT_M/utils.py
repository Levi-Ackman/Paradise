import math
import torch
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_, _no_grad_uniform_
import warnings



class TSAT_parameter():
    """
    The TSAT's parameter is defined here. It contains model parameter and training parameter. Different dataset use different parameter setting.

    :param dataset_name
    """
    def __init__(self, args):
        self.model_parameters, self.training_parameters = None, None
        assert args.dataset != None     # The name of dataset should not be None, please check dataset_name input correctly
        if args.dataset == 'default':
            self.model_parameters = {
            'l_backcast': None,                             # backcast length
            'd_edge': None,                                 # edge features (number of IMF used)
            'd_model': None,                                # model hidden layer
            'N': None,                                      # number of Self_Attention_Block
            'h': None,                                      # Multi-attention heads
            'N_dense': None,                                # Sequential feed forward layers
            'n_output': None,
            'n_nodes':None,
            'lambda_attention': None,
            'lambda_imf_1': None,
            'lambda_imf_2': None,
            'lambda_imf_3': None,
            'lambda_imf_4': None,
            'lambda_imf_5': None,
            'dense_output_nonlinearity': None,
            'imf_matrix_kernel': None,
            'dropout': None,
            'aggregation_type': None,
            'scale_norm': True,
            'trainable_lambda':True
            }
            self.training_parameters = {
                'total_epochs': 5,
                'batch_size': 64,
                'loss_function': 'rmse',
                'metric': 'rmse',
            }
        # =====================================================================================
        # ETTh1_48
        elif args.dataset == 'hand_foot':    #
            self.model_parameters = {
            'l_backcast': args.seq_len,                          # backcast length
            'd_edge': args.d_edge,                           # edge features (number of IMF used)
            'd_model': args.d_model,                         # model hidden layer
            'N': args.n_blocks,                                 # number of Self_Attention_Block
            'h': args.n_heads,                                 # Multi-attention heads
            'N_dense': args.n_dense,                           # Sequential feed forward layers
            'n_output': args.pre_len*7,
            'n_nodes':args.n_node,
            'lambda_attention': args.lambda_attention,
            'lambda_imf_1': args.lambda_imf_1,
            'lambda_imf_2': args.lambda_imf_2,
            'lambda_imf_3': args.lambda_imf_3,
            'lambda_imf_4': args.lambda_imf_4,
            'lambda_imf_5': args.lambda_imf_5,
            'dense_output_nonlinearity': args.dense_activation,
            'imf_matrix_kernel': args.imf_matrix_kernel,
            'dropout': args.dropout_ratio,
            'aggregation_type': args.aggregation_type,
            'scale_norm': args.is_scale_norm,
            'trainable_lambda':args.trainable_lambda
            }
            self.training_parameters = {
                'total_epochs': args.num_epochs,
                'batch_size': args.batch_size,
                'loss_function': args.loss_function,
                'metric': args.metric,
                'device':torch.device('cpu'),
            }

    def parameters(self):
        """
        Return the model parameter and training parameter
        """
        return self.model_parameters, self.training_parameters

    # def creat_dataset_parameter(self, mp, tp):
    #     self.model_parameters = mp
    #     self.training_parameters = tp
    #     return self



def xavier_normal_small_init_(tensor, gain=1.):
    """
    Type: (Tensor, float) -> Tensor
    :param tensor

    :return: tensor
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))
    return _no_grad_normal_(tensor, 0., std)


def xavier_uniform_small_init_(tensor, gain=1.):
    """
    Type: (Tensor, float) -> Tensor
    :param tensor

    :return: tensor
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return _no_grad_uniform_(tensor, -a, a)


def loss_function(loss_function_name:str):
    """
    Define the loss function here
    :param loss_function_name: the name of loss function

    :return: a PyTorch loss function
    """
    assert loss_function_name != None   # the loss function should not be none
    if loss_function_name == 'rmse':
        return torch.nn.MSELoss()
    elif loss_function_name == 'mae':
        return torch.nn.L1Loss()
    elif loss_function_name == 'smoothed mae':
        return torch.nn.SmoothL1Loss()
    elif loss_function_name == 'Cross Entropy Loss':
        return torch.nn.CrossEntropyLoss
    elif loss_function_name == 'Huber Loss':
        return torch.nn.HuberLoss()
    elif loss_function_name == 'bce':
        return torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        assert f'Unsupported loss function name : {loss_function_name}'

def calculate_loss(y_true, y_pred, loss_function_name, criterion, device):
    """
    y_true.shape = (batch, num_tasks)
    y_pred.shape = (batch, num_tasks)
    """
    if loss_function_name == 'mae':
        y_true, y_pred = y_true.flatten(), y_pred.flatten()
        loss = criterion(y_pred, y_true)
    elif loss_function_name == 'smoothed mae':
        y_true, y_pred = y_true.flatten(), y_pred.flatten()
        loss = criterion(y_pred, y_true)
    elif loss_function_name == 'rmse':
        y_true, y_pred = y_true.flatten(), y_pred.flatten()
        loss = torch.sqrt(criterion(y_pred, y_true))
    elif loss_function_name == 'Huber Loss':
        y_true, y_pred = y_true.flatten(), y_pred.flatten()
        loss = criterion(y_pred, y_true)
    elif loss_function_name == 'bce':
        # find all -1 in y_true
        y_true = y_true.long()
        y_mask = torch.where(y_true == -1, torch.tensor([0]).to(device), torch.tensor([1]).to(device))
        y_cal_true = torch.where(y_true == -1, torch.tensor([0]).to(device), y_true).float()
        loss = criterion(y_pred, y_cal_true) * y_mask
        loss = loss.sum() / y_mask.sum()
    else:
        loss = criterion(y_pred, y_true)
    return loss

def evaluation(y_true, y_pred, y_graph_name, requirement, data_mean=0, data_std=1):
    """
    This function is to evaluate the result of y_true and y_pred and calculate the corresponding loss measurement
    
    y_true.shape = (samples, task_numbers)
    y_pred.shape = (samples, task_numbers)

    :return: collect_result: a dict consists of different items, e.g. rmse, sample
    """
    collect_result = {}
    assert len(requirement) != 0    # the requirement should not be empty
    if 'sample' in requirement:
        collect_result['graph_name'] = y_graph_name.tolist()
        collect_result['prediction'] = (y_pred * data_std + data_mean).tolist()
        collect_result['label'] = y_true.tolist()
    if 'rmse' in requirement:
        # y_true, y_pred = y_true.flatten(), (y_pred.flatten() * data_std + data_mean).tolist()
        y_true, y_pred = y_true.flatten(), y_pred.flatten()
        collect_result['rmse'] = np.sqrt(F.mse_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='mean'))
    if 'mse and mae' in requirement:
        # y_true, y_pred = y_true.flatten(), (y_pred.flatten() * data_std + data_mean).tolist()
        collect_result['mse'] = F.mse_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='mean')
        collect_result['mae'] = F.l1_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='mean')
    if 'mae' in requirement:
        y_true, y_pred = y_true.flatten(), (y_pred.flatten() * data_std + data_mean).tolist()
        collect_result['mae'] = F.l1_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='mean')
    if 'auc' in requirement:
        auc_score_list = []
        if y_true.shape[1] > 1:
            for label in range(y_true.shape[1]):
                true, pred = y_true[:, label], y_pred[:, label]
                # all 0's or all 1's
                if len(set(true)) == 1:
                    auc_score_list.append(float('nan'))
                else:
                    auc_score_list.append(metrics.roc_auc_score(true[np.where(true >= 0)], pred[np.where(true >= 0)]))
            collect_result['auc'] = np.nanmean(auc_score_list)
        else:
            collect_result['auc'] = metrics.roc_auc_score(y_true, y_pred)
    if 'bce' in requirement:
        # find all -1 in y_true
        y_mask = np.where(y_true == -1, 0, 1)
        y_cal_true = np.where(y_true == -1, 0, y_true)
        loss = F.binary_cross_entropy_with_logits(torch.tensor(y_pred), torch.tensor(y_cal_true), reduction='none') * y_mask
        collect_result['bce'] = loss.sum() / y_mask.sum()
    # Checking any missing requirement that not yet cover
    for item in requirement:
        if item not in ['sample', 'rmse', 'mse and mae', 'mae', 'auc', 'bce']:
            warnings.warn(f'{item} is not in requirement. Therefore, the collect result does not contain this.')
    return collect_result

import matplotlib.pyplot as plt
def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

import numpy as np

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def R2(pred, true):
    return 1 - np.mean((pred - true) ** 2) / np.var(true)

def metric(pred, true):
    mse = MSE(pred, true)
    mae = MAE(pred, true)
    r_squared = R2(pred, true)

    return mse, mae, r_squared
