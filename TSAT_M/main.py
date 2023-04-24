import argparse
from collections import defaultdict
from dataset_TSAT_ETTm1_48 import generate_hd_data
from dataset_graph import construct_TSAT_dataset, graph_collate_func_TSAT_normalization_require
import numpy as np
from sklearn.model_selection import train_test_split
from TSAT import make_TSAT_model
import time
import torch
from torch.utils.data import DataLoader
from utils import TSAT_parameter, loss_function, calculate_loss
from tqdm import tqdm
import os
import json
from utils import visual,metric

Lr_option=[1e-2,5e-3,1e-3,5e-4,1e-4]
Batch_size_op=[64,48,32,16]
Optim_op=['Adam','SGD','RMSProp']
input_label_pres_opt=[
    [128,32],[128,64],
    [144,64],
    [192,64],[192,128]]


class KOI_model_train_test_interface():
    def __init__(self, args,TSAT_model, model_params:dict, train_params:dict) -> None:
        self.TSAT_model = TSAT_model
        self.TSAT_model = self.TSAT_model.to(train_params['device'])    # send the model to GPU
        self.train_params = train_params
        self.model_params = model_params
        self.learning_rate=args.lr
        self.args=args
        self.criterion = loss_function(train_params['loss_function'])
        if args.optimizer=='Adam':
            self.optimizer = torch.optim.Adam(self.TSAT_model.parameters(),lr=args.lr, betas=(0.999,0.9))
        elif args.optimizer=='RMSprop':
            self.optimizer = torch.optim.RMSprop(self.TSAT_model.parameters(),lr=args.lr)
        elif args.optimizer=='SGD':
            self.optimizer = torch.optim.SGD(self.TSAT_model.parameters(),lr=args.lr)
        
        
        self.num_workers=args.num_workers ## workers to handle the dataloader process


    def import_dataset(self, dataset) -> None:
        # import the dataset
        train_valid_split_ratio = 0.2
        train_valid_dataset, self.test_dataset = train_test_split(dataset, test_size=train_valid_split_ratio, shuffle=True)  # False
        self.train_dataset, self.valid_dataset = train_test_split(train_valid_dataset, test_size=train_valid_split_ratio)
        # Data Loader
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.train_params['batch_size'], 
                                       collate_fn=graph_collate_func_TSAT_normalization_require, shuffle=True,
                                       drop_last=True, num_workers=self.num_workers, pin_memory=False)
        
        self.valid_loader = DataLoader(dataset=self.valid_dataset, batch_size=self.train_params['batch_size'], 
                                       collate_fn=graph_collate_func_TSAT_normalization_require, shuffle=True, 
                                       drop_last=True, num_workers=self.num_workers, pin_memory=False)
        
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.train_params['batch_size'], 
                                       collate_fn=graph_collate_func_TSAT_normalization_require, shuffle=False, 
                                       drop_last=True, num_workers=self.num_workers, pin_memory=False)

    def calculate_number_of_parameter(self) -> int:
        model_parameters = filter(lambda p: p.requires_grad, self.TSAT_model.parameters())
        return int(sum([np.prod(p.size()) for p in model_parameters]))

    def view_train_params(self):
        # view the training parameters
        return self.train_params
    
    def view_model_params(self):
        # view the model parameters
        return self.model_params

    def train_model(self) -> None:
        path = os.path.join(self.args.checkpoints, 'seq_'+str(self.args.seq_len)+'_pre_'+str(self.args.pre_len)+'/'+self.args.model \
            + '/'+'opt_'+self.args.optimizer +'/'+'bz_'+str(self.args.batch_size)+'/')
        if not os.path.exists(path):
            os.makedirs(path)
        # training start
        start_time = time.time()
        self.TSAT_model.train()
        train_total_loss=0.0
        best_val_total_loss=20e+4
        # for epoch in tqdm(range(self.train_params['total_epochs'])):
        for epoch in range(self.train_params['total_epochs']):
            train_epoch_loss=0.0
            for batch in self.train_loader:
                graph_name_list, adjacency_matrix, node_features, imf_1_matrices, imf_2_matrices, imf_3_matrices, imf_4_matrices, imf_5_matrices, y_true_normalization, _, _ = batch
                adjacency_matrix = adjacency_matrix.to(self.train_params['device'])  # (batch, max_length, max_length)
                node_features = node_features.to(self.train_params['device'])  # (batch, max_length, d_node)
                imf_1_matrices = imf_1_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
                imf_2_matrices = imf_2_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
                imf_3_matrices = imf_3_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
                imf_4_matrices = imf_4_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
                imf_5_matrices = imf_5_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
                y_true_normalization = y_true_normalization.to(self.train_params['device'])  # (batch, task_numbers)
                batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch, max_length)
                y_pred_normalization = self.TSAT_model(
                    node_features, batch_mask, adjacency_matrix, imf_1_matrices, imf_2_matrices, imf_3_matrices, imf_4_matrices, imf_5_matrices
                    )
                train_loss = calculate_loss(y_true_normalization, y_pred_normalization, self.train_params['loss_function'], self.criterion, self.train_params['device'])
                train_epoch_loss+=train_loss
                
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
            val_total_loss=self._val_model()
            if best_val_total_loss>val_total_loss:
                best_val_total_loss=val_total_loss
                best_model_path = path+'_lr_'+str(self.args.lr)+ 'checkpoint.pth'
                torch.save(self.TSAT_model.state_dict(),best_model_path)
            
               
            print(' loss of this epoch:{}: {}'.format(epoch,train_epoch_loss/len(self.train_loader)))
            train_total_loss+=train_epoch_loss/len(self.train_loader)
                
        # save model
        end_time = time.time()
        print(f' TSAT train complete! Training time: {end_time-start_time}')
        print(' train_total_loss:{}'.format(train_total_loss/self.train_params['total_epochs']))

        # return self.model
    
    def _val_model(self) -> None:
        # testing start
        start_time = time.time()
        # self.TSAT_model.load_state_dict(torch.load('src/best_model.pt'))
        self.TSAT_model.eval()
        val_total_loss=0.0
        for val_batch in self.valid_loader:
            graph_name_list, adjacency_matrix, node_features, imf_1_matrices, imf_2_matrices, imf_3_matrices, imf_4_matrices, imf_5_matrices, y_true_normalization, _, _ = val_batch
            adjacency_matrix = adjacency_matrix.to(self.train_params['device'])  # (batch, max_length, max_length)
            node_features = node_features.to(self.train_params['device'])  # (batch, max_length, d_node)
            imf_1_matrices = imf_1_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            imf_2_matrices = imf_2_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            imf_3_matrices = imf_3_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            imf_4_matrices = imf_4_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            imf_5_matrices = imf_5_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            y_true_normalization = y_true_normalization.to(self.train_params['device'])  # (batch, task_numbers)
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch, max_length)
            y_pred_normalization = self.TSAT_model(
                node_features, batch_mask, adjacency_matrix, imf_1_matrices, imf_2_matrices, imf_3_matrices, imf_4_matrices, imf_5_matrices
                )
            val_loss = calculate_loss(y_true_normalization, y_pred_normalization, self.train_params['loss_function'], self.criterion, self.train_params['device'])
            # evaluate()
            val_total_loss+=val_loss
            
        end_time = time.time()
        val_total_loss=val_total_loss/len(self.valid_loader)
        
        print(f' TSAT val complete! Testing time: {end_time-start_time}')
        print(' val_total_loss:{}'.format(val_total_loss))
        
        return val_total_loss
        
    def test_model(self) -> None:
        # testing start
        start_time = time.time()
        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/'+'seq_'+str(self.args.seq_len)+'_pre_'+str(self.args.pre_len)+'/'+self.args.model+'/'+'opt_'+self.args.optimizer +'/'+ 'bz_'+str(self.args.batch_size)+'/'+'lr_'+str(self.args.lr)+ '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        self.TSAT_model.eval()
        val_total_loss=0.0
        for val_batch in self.test_loader:
            graph_name_list, adjacency_matrix, node_features, imf_1_matrices, imf_2_matrices, imf_3_matrices, imf_4_matrices, imf_5_matrices, y_true_normalization, _, _ = val_batch
            adjacency_matrix = adjacency_matrix.to(self.train_params['device'])  # (batch, max_length, max_length)
            node_features = node_features.to(self.train_params['device'])  # (batch, max_length, d_node)
            imf_1_matrices = imf_1_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            imf_2_matrices = imf_2_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            imf_3_matrices = imf_3_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            imf_4_matrices = imf_4_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            imf_5_matrices = imf_5_matrices.to(self.train_params['device']) # (batch, max_length, max_length)
            y_true_normalization = y_true_normalization.to(self.train_params['device'])  # (batch, task_numbers)
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch, max_length)
            y_pred_normalization = self.TSAT_model(
                node_features, batch_mask, adjacency_matrix, imf_1_matrices, imf_2_matrices, imf_3_matrices, imf_4_matrices, imf_5_matrices
                )
            val_loss = calculate_loss(y_true_normalization, y_pred_normalization, self.train_params['loss_function'], self.criterion, self.train_params['device'])
            # evaluate()
            val_total_loss+=val_loss
            
            idx=8
            pred = y_pred_normalization.detach().cpu().numpy()
            true = y_true_normalization.detach().cpu().numpy()
            preds.append(pred)
            trues.append(true)
        preds = np.array(preds)
        trues = np.array(trues)
        visual(true[-1,:], pred[-1,:], os.path.join(folder_path, str(idx) + '.pdf'))

        end_time = time.time()
        print(f' TSAT test complete! Testing time: {end_time-start_time}')
        print(' test_total_loss:{}'.format(val_total_loss/len(self.valid_loader)))
        
        mse, mae, r_squared = metric(preds, trues)
        print('mse:{:.4f}, mae:{:.4f}, R2:{:.4f}'.format(mse/len(self.valid_loader), mae/len(self.valid_loader), r_squared/len(self.valid_loader)))
        my_dict = {
            'model':self.args.model,
            'batch_size':str(self.args.batch_size),
            'lr':str(self.args.lr),
            'Optim':self.args.optimizer,
            'mse':float(mse), 
            'mae':float(mae), 
            'R2':float(r_squared)}
        dict_path='./test_dict/'+'seq_'+str(self.args.seq_len)+'_pre_'+str(self.args.pre_len)+'/'+self.args.model+'/'+'opt_'+self.args.optimizer+'/'+'bz_'+str(self.args.batch_size)+'/'+'lr_'+str(self.args.lr)
        if not os.path.exists(dict_path):
            os.makedirs(dict_path)
        with open(os.path.join(dict_path, 'records.json'), 'w') as f:
             json.dump(my_dict, f)
        f.close()



if __name__ == '__main__':
    ## init args
    lr=8e-3
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False, default='TSAT',help='model name, options: [ TSAT ]')
    parser.add_argument("--gpu", type=str, help='gpu', default=0)
    parser.add_argument("--pre_len", type=int, help='pre_len', default=16)
    parser.add_argument("--seq_len", type=int, help='input_seq_len', default=64)
    parser.add_argument("--correlation", type=str, help='mat_adj_op:[fully_connected,correlation,zero_mat,random]', default='correlation')
    parser.add_argument("--use_tqdm", type=bool, help='is_use_tqdm', default=False)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    
    parser.add_argument("--batch_size", type=int, help='batch_size', default=64)
    parser.add_argument("--num_workers", type=int, help='num_workers', default=0)
    parser.add_argument("--num_epochs", type=int, help='num_epochs', default=3000)
    parser.add_argument("--loss_function", type=str, help='loss_function', default='rmse')
    parser.add_argument("--metric", type=str, help='evaluation_metric', default='rmse')
    parser.add_argument("--dropout_ratio", type=float, help='dropout_ratio', default=0.05)
    
    parser.add_argument("--dataset", type=str, help='name of dataset', default='hand_foot')
    
    parser.add_argument("--d_model", type=int, help='hidden_dim_of_model', default=2048)
    parser.add_argument("--n_dense", type=int, help='Sequential feed forward layers', default=1)
    parser.add_argument("--n_heads", type=int, help='Multi-attention heads', default=8)
    parser.add_argument("--n_blocks", type=int, help='number of Self_Attention_Block', default=8)
    parser.add_argument("--d_edge", type=int, help='edge features (number of IMF used)', default=5)
    parser.add_argument("--n_node", type=int, help='default_node_num(num_of_multivariates)', default=7)
    parser.add_argument("--dense_activation", type=str, help='dense_output_nonlinearity:[relu,tanh]', default='relu')
    parser.add_argument("--aggregation_type", type=str, help='graph data aggregation_type:[mean,splitted_mean,sum,dummy_node]', default='mean')
    parser.add_argument("--imf_matrix_kernel", type=str, help='imf_matrix_kernel:[exp,softmax,identical,absolute]', default='softmax')
    parser.add_argument("--lambda_attention", type=float, help='lambda_attention', default=0.33)
    parser.add_argument("--lambda_imf_1", type=float, help='lambda_imf_1', default=0.0999)
    parser.add_argument("--lambda_imf_2", type=float, help='lambda_imf_2', default=0.0801)
    parser.add_argument("--lambda_imf_3", type=float, help='lambda_imf_3', default=0.06)
    parser.add_argument("--lambda_imf_4", type=float, help='lambda_imf_4', default=0.0399)
    parser.add_argument("--lambda_imf_5", type=float, help='lambda_imf_5', default= 0.0201)
    
    parser.add_argument("--trainable_lambda", type=bool, help='trainable_lambda_choice', default=True)
    parser.add_argument("--is_scale_norm", type=bool, help='trainable_lambda_choice', default=False)
    
    parser.add_argument("--lr", type=float, help='lr_choice', default=1e-2)
    parser.add_argument("--optimizer", type=str, help='optim_op', default='Adam')
    
    args = parser.parse_args()
    print(args)
    TSAT_parameters = TSAT_parameter(args)
    model_params, train_params = TSAT_parameters.parameters()

    ## Check GPU is available
    train_params['device'] = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
    print('N_GPU:{}'.format(torch.device(0)))
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    # generate graph from data
    '''personal_data'''
    data_graph, data_label = generate_hd_data(
        n_lookback_days=args.seq_len, 
        n_lookforward_days=args.pre_len, 
        adj_mat_method='correlation', 
        use_tqdm=True,
        )
    dataset = construct_TSAT_dataset(data_graph, data_label, normalization_require=True)
    total_metrics = defaultdict(list)

    ## main train and test
    TSAT_model = make_TSAT_model(**model_params)
    model_interface = KOI_model_train_test_interface(args,TSAT_model, model_params, train_params)
    model_interface.import_dataset(dataset=dataset)
    model_interface.train_model()
    model_interface.test_model()
