import pandas as pd
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import seaborn as sns
import soundfile as sf
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import torch.utils.data as data_utils
import torch.optim as optim
import random
from tqdm import tqdm
import pickle
from transformers import BertModel, BertTokenizer
from IPython.display import FileLink

class model_register():
    def __init__(self, ):
        self.batch_size = 50 #1024
        self.loss_function = nn.CrossEntropyLoss()
        self.lr = 1e-3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        self.epoch = 0
        
	#features_train, features_test, targets_train, targets_test
    def gen_datasets(self, X_train, X_test, y_train, y_test,  mode='spectr'):
        #X_train, X_test, y_train, y_test = train_test_split(features[mode], pd.DataFrame(targets).to_numpy(), test_size=0.2, random_state=0)
        X_train = X_train[mode]
        train_id = np.array([int(i.split('|')[0]) for i in pd.DataFrame(y_train)['id'].to_list()]).astype(np.int64)
        y_train = pd.DataFrame(y_train)[['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear', 'sentiment']].to_numpy()
        
        X_test = X_test[mode]
        test_id = np.array([int(i.split('|')[0]) for i in pd.DataFrame(y_test)['id'].to_list()]).astype(np.int64)
        y_test = pd.DataFrame(y_test)[['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear', 'sentiment']].to_numpy()
        y_test = np.nan_to_num(y_test, nan=0)
        y_train = np.nan_to_num(y_train, nan=0)
        X_train = np.nan_to_num(X_train, nan=0)
        X_test = np.nan_to_num(X_test, nan=0)

        inputs_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)

        targets_train_emo = torch.tensor([i[:-1] for i in y_train], dtype=torch.long)
        targets_train_sent = torch.tensor([i[-1] for i in y_train], dtype=torch.long)
        train_id = torch.tensor(train_id, dtype=torch.long)
        
        inputs_test = torch.tensor(X_test, dtype=torch.float32)
        targets_test_emo = torch.tensor([i[:-1] for i in y_test], dtype=torch.long)
        targets_test_sent = torch.tensor([i[-1] for i in y_test], dtype=torch.long)
        test_id = torch.tensor(test_id, dtype=torch.long)
        
        self.input_dim = inputs_train.shape[1:]
        self.input_dim = torch.prod(torch.tensor(self.input_dim))
        
        inputs_train = inputs_train.view(inputs_train.shape[0], self.input_dim)
        inputs_test = inputs_test.view(inputs_test.shape[0], self.input_dim)
        
        train = data_utils.TensorDataset(
                                     inputs_train.to(self.device), 
                                     targets_train_emo.to(self.device), 
                                     targets_train_sent.to(self.device),
                                     train_id.to(self.device)
                                     )
        test = data_utils.TensorDataset(
                                     inputs_test.to(self.device), 
                                     targets_test_emo.to(self.device), 
                                     targets_test_sent.to(self.device),
                                     test_id.to(self.device)
                                     )
        
        self.trainset = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=True)
        self.testset = torch.utils.data.DataLoader(test, batch_size=self.batch_size, shuffle=False)

    def get_model(self, Classifier, addition=None, mode='standart'):
        if mode=='standart':
            self.model = Classifier(self.input_dim, num_emo_classes=2, num_sent_classes=3).to(self.device)
        elif mode == 'add':
            self.model = Classifier(addition, num_emo_classes=2, num_sent_classes=3).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
    
    def agrigation(self, emo_set, sent=False):
        if len(emo_set) == 0:
            return []
        count = Counter(emo_set)
        max_freq = max(count.values())
        most_common_nums = [num for num, freq in count.items() if freq == max_freq]
        if sent:
            if 2 in most_common_nums:
                return 2
            elif 3 in most_common_nums:
                return 3
            elif 4 in most_common_nums:
                return 4
            elif 1 in most_common_nums:
                return 1
            elif 0 in most_common_nums:
                return 0
            else:
                return most_common_nums[0]
            
        return min(most_common_nums)

    def train(self, epochs=10, testing=True, count_zero=False, tensor_type='float'):
        for epoch in range(epochs):
            
            with tqdm(self.trainset, desc=f"Epoch {epoch+1}/{epochs}", leave=True) as pbar:
                for X, y_emo, y_sent, id_list in pbar:
                    X, y_emo, y_sent = X, y_emo, y_sent
                    if tensor_type == 'long':
                        X = X.long()

                    self.optimizer.zero_grad()
                    emo_out, sent_out = self.model(X)
                    emo_out = emo_out.view(-1, emo_out.shape[-1])
                    y_emo = y_emo.view(-1) 
                    loss_emo = self.loss_function(emo_out, y_emo)
                    loss_sent = self.loss_function(sent_out, y_sent)
                    loss = loss_emo + loss_sent
        
                    loss.backward()
                    self.optimizer.step()
                    
                    pbar.set_postfix(loss=loss.item(), loss_emo=loss_emo.item(), loss_sent=loss_sent.item())
            self.scheduler.step()
            self.epoch += 1
            if testing:
                self.test(to_print=False, epoch=self.epoch, count_zero=count_zero, tensor_type=tensor_type)

    def test(self, to_print=True, epoch='-', count_zero=True, tensor_type='float'):
        self.model.eval()
        predictions_emo, targets_emo = [], []
        predictions_sent, targets_sent = [], []
        ids_list = []
        self.results[epoch] = {}
        if count_zero == True:
            labels_list = [0, 1, 2, 3]
        else:
            labels_list = [1, 2, 3]
        
        with torch.no_grad():
            with tqdm(self.testset, desc="Testing", leave=True) as pbar:
                for X, y_emo, y_sent, id_list in pbar:
                    X, y_emo, y_sent, id_list = X.to(self.device), y_emo.cpu(), y_sent.cpu(), id_list.cpu()

                    emo_out, sent_out = self.model(X)
                    
                    preds_emo = torch.argmax(emo_out, dim=-1).cpu().numpy()
                    preds_sent = torch.argmax(sent_out, dim=-1).cpu().numpy()
                    
                    targets_emo.extend(y_emo.numpy())
                    targets_sent.extend(y_sent.numpy())
                    predictions_emo.extend(preds_emo)
                    predictions_sent.extend(preds_sent)
                    ids_list.extend(id_list)
    
        emos = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']
        predictions_per_emo = {emo: [] for emo in emos}
        targets_per_emo = {emo: [] for emo in emos}
    
        i_last = None
        for h, t, i in zip(predictions_emo, targets_emo, ids_list):
            if i_last == None:
                i_last = i
                sub_pred_per_emo = {emo: [] for emo in emos}
                sub_targ_per_emo = {emo: [] for emo in emos}
            elif i_last != i:
                for emo in emos:
                    predictions_per_emo[emo].append(self.agrigation(sub_pred_per_emo[emo]))
                    targets_per_emo[emo].append(self.agrigation(sub_targ_per_emo[emo]))
                sub_pred_per_emo = {emo: [] for emo in emos}
                sub_targ_per_emo = {emo: [] for emo in emos}
            for n, emo in enumerate(emos):
                if t[n] != 0 or count_zero:  # not to count 0 if count_zero=False
                    sub_pred_per_emo[emo].append(h[n])
                    sub_targ_per_emo[emo].append(t[n])
            i_last = i
        for emo in emos:
            predictions_per_emo[emo].append(self.agrigation(sub_pred_per_emo[emo]))
            targets_per_emo[emo].append(self.agrigation(sub_targ_per_emo[emo]))
        for emo in emos:
            print('EMO:', emo)
            print('Was dist: ', dict(sorted(Counter(predictions_per_emo[emo]).items())))
            print('Dist emo: ', dict(sorted(Counter(predictions_per_emo[emo]).items())))
            print('Target d: ', dict(sorted(Counter(targets_per_emo[emo]).items())))

        predictions_sent_upd = []
        targets_sent_upd = []
        i_last = None
        for h, t, i in zip(predictions_sent, targets_sent, ids_list):
            if i_last == None:
                i_last = i
                sub_pred_sent = []
                sub_targ_sent = []
            elif i_last != i:
                predictions_sent_upd.append(self.agrigation(sub_pred_sent, sent=True))
                targets_sent_upd.append(self.agrigation(sub_targ_sent, sent=True))
                sub_pred_sent = []
                sub_targ_sent = []
            sub_pred_sent.append(h)
            sub_targ_sent.append(t)
            i_last = i
        predictions_sent_upd.append(self.agrigation(sub_pred_sent, sent=True))
        targets_sent_upd.append(self.agrigation(sub_targ_sent, sent=True))
        print('SENT')
        print('Was dist: ', dict(sorted(Counter(predictions_sent).items())))
        print('Dist emo: ', dict(sorted(Counter(predictions_sent_upd).items())))
        print('Target d: ', dict(sorted(Counter(targets_sent_upd).items())))
        predictions_sent = predictions_sent_upd
        targets_sent = targets_sent_upd
        text = ''
        # F1-Score 
        mean_f1 = []
        for emo in emos:
            f1_value = round(f1_score(targets_per_emo[emo], predictions_per_emo[emo], 
                                      average="macro", zero_division=0, labels=labels_list) * 100, 2)
            mean_f1.append(f1_value)
            self.results[epoch][emo.capitalize()] = {'f1_score': f1_value}
            if to_print:
                stext = f'F1 {emo.capitalize():10}: {f1_value}'
                print(stext)
                text += '\n' + stext
        if to_print:
            mean_f1 = round(np.mean(mean_f1), 2)
            stext = f'F1 mean        : {mean_f1}'
            print(stext)
            text += '\n' + stext
        
        f1_sentiment = round(f1_score(targets_sent, predictions_sent, average="macro", 
                                      zero_division=0, labels=labels_list) * 100, 2)
        self.results[epoch]['Sentiment'] = {'f1_score': f1_sentiment}
        if to_print:
            stext = f'F1 Sentiment   : {f1_sentiment}'
            print(stext)
            text += '\n' + stext
    
        # RMSE 
        mean_rmse = []
        for emo in emos:
            rmse_value = round(root_mean_squared_error(targets_per_emo[emo], predictions_per_emo[emo]), 2)
            self.results[epoch][emo.capitalize()]['rmse'] = rmse_value
            mean_rmse.append(rmse_value)
            if to_print:
                stext = f'RMSE {emo.capitalize():10}: {rmse_value}'
                print(stext)
                text += '\n' + stext
        if to_print:
            mean_rmse = round(np.mean(mean_rmse), 2)
            stext = f'RMSE mean      : {mean_rmse}'
            print(stext)
            text += '\n' + stext
        
        rmse_sentiment = round(root_mean_squared_error(targets_sent, predictions_sent), 2)
        self.results[epoch]['Sentiment']['rmse'] = rmse_sentiment
        if to_print:
            stext = f'RMSE Sentiment : {rmse_sentiment}'
            print(stext)
            text += '\n' + stext
        self.text = text


class classifier_v2(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=4, nheads=8, num_emo=6, num_emo_classes=4, num_sent_classes=5, dropout_rate=0.3):
        super().__init__()

        self.num_emo = num_emo
        self.num_emo_classes = num_emo_classes
        self.hidden_dim = hidden_dim*8

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads, dim_feedforward=hidden_dim * 4, dropout=dropout_rate)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_emo = nn.Linear(hidden_dim, num_emo * num_emo_classes)  
        self.fc_sent = nn.Linear(hidden_dim, num_sent_classes)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = x.unsqueeze(0) 
        
        x = self.transformer(x) 
        x = x.squeeze(0)
        
        x = self.dropout(x)

        emo_out = self.fc_emo(x).view(-1, self.num_emo, self.num_emo_classes)

        sent_out = self.fc_sent(x)
        return emo_out, sent_out
        
class classifier_v1(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_emo=6, num_emo_classes=4, num_sent_classes=5, dropout_rate=0.1):
        super().__init__()
        hidden_dim = hidden_dim * 4

        self.num_emo = num_emo
        self.num_emo_classes = num_emo_classes

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_emo = nn.Linear(hidden_dim, num_emo * num_emo_classes)
        self.fc_sent = nn.Linear(hidden_dim, num_sent_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        emo_out = self.fc_emo(x).view(-1, self.num_emo, self.num_emo_classes)
        sent_out = self.fc_sent(x)

        return emo_out, sent_out

class classifier_v1_token(nn.Module):
    def __init__(self, input_dim, embedding_dim=128, hidden_dim=512, num_emo=6, num_emo_classes=4, num_sent_classes=5, dropout_rate=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        hidden_dim = hidden_dim * 4

        self.num_emo = num_emo
        self.num_emo_classes = num_emo_classes

        # Первый слой теперь - это слой эмбеддинга
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc_emo = nn.Linear(hidden_dim, num_emo * num_emo_classes)
        self.fc_sent = nn.Linear(hidden_dim, num_sent_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embedding(x)
        
        x = x.mean(dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        emo_out = self.fc_emo(x).view(-1, self.num_emo, self.num_emo_classes)
        sent_out = self.fc_sent(x)

        return emo_out, sent_out

class classifier_v1_bert(nn.Module):
    def __init__(self, bert_model_name, num_emo=6, num_emo_classes=4, num_sent_classes=5, dropout_rate=0.5):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_output_dim = self.bert.config.hidden_size
        
        self.num_emo = num_emo
        self.num_emo_classes = num_emo_classes

        self.fc1 = nn.Linear(self.bert_output_dim, 512)
        self.fc_emo = nn.Linear(512, num_emo * num_emo_classes)
        self.fc_sent = nn.Linear(512, num_sent_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        attention_mask = (x != 0).long()
        x = x.long()
        outputs = self.bert(x, attention_mask=attention_mask)
        x = outputs.last_hidden_state
        x = x.mean(dim=1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        emo_out = self.fc_emo(x).view(-1, self.num_emo, self.num_emo_classes)
        sent_out = self.fc_sent(x)

        return emo_out, sent_out


class classifier_v2_token(nn.Module):
    def __init__(self, input_dim, embedding_dim=128, hidden_dim=512, num_emo=6, num_emo_classes=4, num_sent_classes=5, dropout_rate=0.5):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        hidden_dim = 128 #hidden_dim

        self.num_emo = num_emo
        self.num_emo_classes = num_emo_classes

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc_emo = nn.Linear(hidden_dim, num_emo * num_emo_classes)
        self.fc_sent = nn.Linear(hidden_dim, num_sent_classes)
        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)

        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        emo_out = self.fc_emo(x).view(-1, self.num_emo, self.num_emo_classes)
        sent_out = self.fc_sent(x)

        return emo_out, sent_out

class classifier_v2_bert(nn.Module):
    def __init__(self, bert_model_name, num_emo=6, num_emo_classes=4, num_sent_classes=5, dropout_rate=0.5):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_output_dim = self.bert.config.hidden_size

        self.num_emo = num_emo
        self.num_emo_classes = num_emo_classes

        self.fc1 = nn.Linear(self.bert_output_dim, 512)
        self.fc_emo = nn.Linear(512, num_emo * num_emo_classes)
        self.fc_sent = nn.Linear(512, num_sent_classes)
        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)

    def forward(self, x):
        attention_mask = (x != 0).long()
        x = x.long()
        outputs = self.bert(x, attention_mask=attention_mask)
        x = outputs.last_hidden_state
        x = x.mean(dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)

        emo_out = self.fc_emo(x).view(-1, self.num_emo, self.num_emo_classes)
        sent_out = self.fc_sent(x)

        return emo_out, sent_out
