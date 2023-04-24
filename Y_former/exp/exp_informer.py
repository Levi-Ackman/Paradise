from data.data_loader import Dataset_ETT_minute,hand_foot
from exp.exp_basic import Exp_Basic
from models.model import Informer, Yformer, Yformer_skipless
from utils.tools import EarlyStopping, adjust_learning_rate,visual
from utils.metrics import metric
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

models= ['yformer','informer','yformer_skipless']

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'yformer':Yformer,
            'yformer_skipless': Yformer_skipless

        }
        if self.args.model=='informer'or self.args.model=="yformer" or self.args.model=="yformer_skipless":
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'hand_foot':hand_foot,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            use_decoder_tokens=args.use_decoder_tokens,
            timeenc=timeenc,
            freq=freq
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.optimizer=='Adam':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        elif self.args.optimizer=='SGD':
            model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer=='RMSprop':
            model_optim = optim.RMSprop(self.model.parameters(), lr=self.args.learning_rate)
        
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()

            if self.args.use_decoder_tokens:
                dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
            else:
                dec_inp = dec_inp.float().to(self.device)

            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, -self.args.pred_len:, :]
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, -self.args.pred_len:, :]
            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true) 

            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
        
    def train(self, setting):
        
        path = os.path.join(self.args.checkpoints, 'seq_'+str(self.args.seq_len)+'_label_'+str(self.args.label_len)+'_pre_'+str(self.args.pred_len)+'/'+self.args.model \
            + '/'+'opt_'+self.args.optimizer +'/'+'bz_'+str(self.args.batch_size)+'/')
        if not os.path.exists(path):
            os.makedirs(path)
            
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
        #     summary(self.model,  [batch_x.shape, batch_x_mark.shape, batch_y.shape, batch_y_mark.shape]) # show the size 
        #     break

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            auto_train_loss = []
            combined_train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
                
                if self.args.use_decoder_tokens:
                    dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = dec_inp.float().to(self.device)
                    
                    
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features=='MS' else 0
                batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
                auto_loss = criterion(outputs[:, :-self.args.pred_len,:], batch_x)
                auto_train_loss.append(auto_loss.item())
                loss = criterion(outputs[:, -self.args.pred_len:,:], batch_y)
                train_loss.append(loss.item())

                combined_loss = self.args.alpha * auto_loss + (1-self.args.alpha) * loss
                combined_train_loss.append(combined_loss.item())
                
                if (i+1) % 600==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | auto loss: {3:.7f} | comb loss: {4:.7f}".format(i + 1, epoch + 1, loss.item(), auto_loss.item(), combined_loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                combined_loss.backward()
                model_optim.step()
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            auto_loss = np.average(auto_train_loss)
            combined_loss = np.average(combined_train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Auto Loss : {3:.7f} | Comb Loss : {4:.7f}, Vali Loss: {5:.7f} Test Loss: {6:.7f}".format(
                epoch + 1, train_steps, train_loss, auto_loss, combined_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting, test=0):
        _, test_loader = self._get_data(flag='test')
        path=os.path.join(self.args.checkpoints, 'seq_'+str(self.args.seq_len)+'_label_'+str(self.args.label_len)+'_pre_'+str(self.args.pred_len)+'/'+self.args.model)
        best_model_path = path + '/' +'opt_'+self.args.optimizer+'/'+'bz_'+str(self.args.batch_size)+'_lr_'+str(self.args.learning_rate)+ 'checkpoint.pth'

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/'+'seq_'+str(self.args.seq_len)+'_label_'+str(self.args.label_len)+'_pre_'+str(self.args.pred_len)+'/'+self.args.model+'/'+'opt_'+self.args.optimizer +'/'+ 'bz_'+str(self.args.batch_size)+'/'+'lr_'+str(self.args.learning_rate)+ '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
                
                if self.args.use_decoder_tokens:
                    dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = dec_inp.float().to(self.device)
                    
                    
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                if idx % self.args.seg == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(idx) + '.pdf'))
        
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        mse, mae, r_squared = metric(preds, trues)
        print('mse:{:.4f}, mae:{:.4f}, R2:{:.4f}'.format(mse, mae, r_squared))
        my_dict = {
            'model':self.args.model,
            'batch_size':str(self.args.batch_size),
            'lr':str(self.args.learning_rate),
            'Optim':self.args.optimizer,
            'mse':float(mse), 
            'mae':float(mae), 
            'R2':float(r_squared)}
        dict_path='./test_dict/'+'seq_'+str(self.args.seq_len)+'_label_'+str(self.args.label_len)+'_pre_'+str(self.args.pred_len)+'/'+self.args.model+'/'+'opt_'+self.args.optimizer+'/'+'bz_'+str(self.args.batch_size)+'/'+'lr_'+str(self.args.learning_rate)
        if not os.path.exists(dict_path):
            os.makedirs(dict_path)
        with open(os.path.join(dict_path, 'records.json'), 'w') as f:
             json.dump(my_dict, f)
        f.close()

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).float()
            if self.args.use_decoder_tokens:
                dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
            else:
                dec_inp = dec_inp.float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
            
            pred = outputs.detach().cpu().numpy()#.squeeze()
            
            preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return