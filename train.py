import pandas as pd
import os
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
from IPython.display import FileLink

ex_dir = '/home/dutov@ad.speechpro.com/work/course/mm_sentiment/'
window = 2
overlap = 0.5
features_train_file = 'ex/'+f'features_train_w{window}_o{overlap}.pickle'
targets_train_file = 'ex/'+f'targets_train_w{window}_o{overlap}.pickle'
features_test_file = 'ex/'+f'features_test_w{window}_o{overlap}.pickle'
targets_test_file = 'ex/'+f'targets_test_w{window}_o{overlap}.pickle'
mode = 'spectr' #mfcc #mel #spectr
num_epochs = 50
name_model = f'models/test_{mode}_{num_epochs}.chkp'

with open(ex_dir+'/'+features_train_file, 'rb') as f:
    features_train = pickle.load(f)

with open(ex_dir+'/'+features_test_file, 'rb') as f:
    features_test = pickle.load(f)

with open(ex_dir+'/'+targets_train_file, 'rb') as f:
    targets_train = pickle.load(f)

with open(ex_dir+'/'+targets_test_file, 'rb') as f:
    targets_test = pickle.load(f)

def set_seed(random_rate=42):
    random.seed(random_rate)
    np.random.seed(random_rate)
    torch.manual_seed(random_rate)
    torch.cuda.manual_seed(random_rate)
    torch.cuda.manual_seed_all(random_rate)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class model_register():
    def __init__(self, ):
        self.batch_size = 64
        self.loss_function = nn.CrossEntropyLoss()
        self.lr = 1e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        self.epoch = 0
        
	#features_train, features_test, targets_train, targets_test
    def gen_datasets(self, X_train, X_test, y_train, y_test,  mode='spectr'):
        #X_train, X_test, y_train, y_test = train_test_split(features[mode], pd.DataFrame(targets).to_numpy(), test_size=0.2, random_state=0)
        X_train = X_train[mode]
        X_test = X_test[mode]
        y_train = pd.DataFrame(y_train).to_numpy()
        y_test = pd.DataFrame(y_test).to_numpy()
        inputs_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        targets_train_emo = torch.tensor([i[:-1] for i in y_train], dtype=torch.long)
        targets_train_sent = torch.tensor([i[-1] for i in y_train], dtype=torch.long)
        inputs_test = torch.tensor(X_test, dtype=torch.float32)
        targets_test_emo = torch.tensor([i[:-1] for i in y_test], dtype=torch.long)
        targets_test_sent = torch.tensor([i[-1] for i in y_test], dtype=torch.long)
        self.input_dim = inputs_train.shape[1:]
        self.input_dim = torch.prod(torch.tensor(model.input_dim))
        inputs_train = inputs_train.view(inputs_train.shape[0], self.input_dim)
        inputs_test = inputs_test.view(inputs_test.shape[0], self.input_dim)
        train = data_utils.TensorDataset(inputs_train.to(self.device), targets_train_emo.to(self.device), targets_train_sent.to(self.device))
        test = data_utils.TensorDataset(inputs_test.to(self.device), targets_test_emo.to(self.device), targets_test_sent.to(self.device))
        
        self.trainset = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=True)
        self.testset = torch.utils.data.DataLoader(test, batch_size=self.batch_size, shuffle=False)

    def get_model(self):
        self.model = Classifier(self.input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, epochs=10, testing=True, count_zero=False):
        for epoch in range(epochs):
            
            with tqdm(self.trainset, desc=f"Epoch {epoch+1}/{epochs}", leave=True) as pbar:
                for X, y_emo, y_sent in pbar: 
                    X, y_emo, y_sent = X, y_emo, y_sent
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
            self.epoch += 1
            if testing:
                self.test(to_print=False, epoch=self.epoch, count_zero=count_zero)

    def test(self, to_print=True, epoch='-', count_zero=True):
        self.model.eval()
        predictions_emo, targets_emo = [], []
        predictions_sent, targets_sent = [], []
        self.results[epoch] = {}
        
        with torch.no_grad():
            with tqdm(self.testset, desc="Testing", leave=True) as pbar:
                for X, y_emo, y_sent in pbar:
                    X, y_emo, y_sent = X.to(self.device), y_emo.cpu(), y_sent.cpu()
    
                    emo_out, sent_out = self.model(X)
                    
                    # Предсказания
                    preds_emo = torch.argmax(emo_out, dim=-1).cpu().numpy()
                    preds_sent = torch.argmax(sent_out, dim=-1).cpu().numpy()
                    
                    # Сохранение истинных значений
                    targets_emo.extend(y_emo.numpy())
                    targets_sent.extend(y_sent.numpy())
                    predictions_emo.extend(preds_emo)
                    predictions_sent.extend(preds_sent)
    
        emos = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']
        predictions_per_emo = {emo: [] for emo in emos}
        targets_per_emo = {emo: [] for emo in emos}
    
        # Обрабатываем эмоции
        for h, t in zip(predictions_emo, targets_emo):
            for i, emo in enumerate(emos):
                if t[i] != 0 or count_zero:  # исключаем метку 0, если count_zero=False
                    predictions_per_emo[emo].append(h[i])
                    targets_per_emo[emo].append(t[i])
    
        # Обрабатываем сентимент
        predictions_sent_upd = []
        targets_sent_upd = []
        for h, t in zip(predictions_sent, targets_sent):
            predictions_sent_upd.append(h)
            targets_sent_upd.append(t)
        predictions_sent = predictions_sent_upd
        targets_sent = targets_sent_upd
    
        text = ''
        # F1-Score для эмоций и сентимента
        for emo in emos:
            f1_value = round(f1_score(targets_per_emo[emo], predictions_per_emo[emo], average="weighted", zero_division=0) * 100, 2)
            self.results[epoch][emo.capitalize()] = {'f1_score': f1_value}
            if to_print:
                stext = f'F1 {emo.capitalize():10}: {f1_value}'
                print(stext)
                text += '\n' + stext
        
        f1_sentiment = round(f1_score(targets_sent, predictions_sent, average="weighted", zero_division=0) * 100, 2)
        self.results[epoch]['Sentiment'] = {'f1_score': f1_sentiment}
        if to_print:
            stext = f'F1 Sentiment   : {f1_sentiment}'
            print(stext)
            text += '\n' + stext
    
        # RMSE для эмоций и сентимента
        for emo in emos:
            rmse_value = round(root_mean_squared_error(targets_per_emo[emo], predictions_per_emo[emo]), 2)
            self.results[epoch][emo.capitalize()]['rmse'] = rmse_value
            if to_print:
                stext = f'RMSE {emo.capitalize():10}: {rmse_value}'
                print(stext)
                text += '\n' + stext
        
        rmse_sentiment = round(root_mean_squared_error(targets_sent, predictions_sent), 2)
        self.results[epoch]['Sentiment']['rmse'] = rmse_sentiment
        if to_print:
            stext = f'RMSE Sentiment : {rmse_sentiment}'
            print(stext)
            text += '\n' + stext
        self.text = text

class Classifier(nn.Module):
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

model = model_register()
model.gen_datasets(features_train, features_test, targets_train, targets_test, mode=mode)
model.get_model()
model.train(num_epochs, testing=False)
model.test(to_print=True, epoch='-', count_zero=False)
with open(name_model.replace('chkp', 'metrics'), 'w') as f:
    f.write(model.text)

torch.save({
            'model_state_dict': model.model.state_dict(),
            }, name_model)
