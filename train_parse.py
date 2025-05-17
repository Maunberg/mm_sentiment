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
from IPython.display import FileLink
import argparse
import torch
torch.cuda.empty_cache()
from utils import model_register
from utils import classifier_v1, classifier_v1_token, classifier_v1_bert, classifier_v3, classifier_v3t3, classifier_v1t3_token
from utils import classifier_v2, classifier_v2_token, classifier_v2_bert, classifier_v1t3, classifier_v3t3_token

parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--ex_dir', type=str, default='/home/dutov@ad.speechpro.com/work/course/mm_sentiment/', help='Директория для экспериментов')
parser.add_argument('--features_type', type=str, 
    choices=['text', 'voice', 'voice_text'],   #MAIN
    default='voice', help='Type of features')
parser.add_argument('--version', type=str, 
    choices=['FNN', 'T', 'RES', 'RES_FNN', 'RES_RES'], default='T',  #MAIN
    help='Model version')
parser.add_argument('--train', type=bool, default=True, help='Is model training')
parser.add_argument('--load_model', type=bool, default=False, help='Is model loading')
parser.add_argument('--num_epochs', type=int, default=25, help='Num epochs')
parser.add_argument('--window', type=str, default=1, help='Audio window')
parser.add_argument('--overlap', type=str, default=0, help='Window overlap')
parser.add_argument('--overlap_test', type=str, default=None, help='Window overlap during testin')
parser.add_argument('--mode', type=str, default='mel', 
    choices=['mel', 'mfcc', 'spectr', 'tfidf', 'token', 'bert', 'audio_text'], 
    help='Type of the modal features')
parser.add_argument('--vocab_size', type=int, default=1155, help='Type of the modal features')
parser.add_argument('--bertmodel', type=str, default='DeepPavlov/rubert-base-cased', help='Bert model name')
parser.add_argument('--n', type=int, default=0, help='N of the model (text)')

args = parser.parse_args()

ex_dir = args.ex_dir
features_type = args.features_type
version = args.version
train = args.train
if train == False:
    load_model = True
else:
    load_model = args.load_model
num_epochs = args.num_epochs
window = args.window
overlap = args.overlap
bertmodel = args.bertmodel
if args.overlap_test == None:
    overlap_test = overlap
else:
    overlap_test = args.overlap_test
mode = args.mode
vocab_size = args.vocab_size
n = args.n
#train = False


if features_type == 'voice':
    features_train_file = 'ex/'+f'features_train_w{window}_o{overlap}.pickle'
    targets_train_file = 'ex/'+f'targets_train_w{window}_o{overlap}.pickle'
    features_test_file = 'ex/'+f'features_test_w{window}_o{overlap_test}.pickle'
    targets_test_file = 'ex/'+f'targets_test_w{window}_o{overlap_test}.pickle'
    name_model = f'models/model_v{version}_m{mode}_w{window}_o{overlap}_e{num_epochs}_voice.chkp'
    metrci_name =  f'models/model_v{version}_m{mode}_w{window}_o{overlap_test}_e{num_epochs}_voice.metric'
elif features_type == 'text':
    features_train_file = 'ex/'+f'features_train_{n}.pickle'
    targets_train_file = 'ex/'+f'targets_train_{n}.pickle'
    features_test_file = 'ex/'+f'features_test_{n}.pickle'
    targets_test_file = 'ex/'+f'targets_test_{n}.pickle'
    name_model = f'models/model_v{version}_{n}_m{mode}_e{num_epochs}_text.chkp'
    metrci_name =  f'models/model_v{version}_{n}_m{mode}_e{num_epochs}_text.metric'
elif features_type == 'voice_text':
    features_train_file = 'ex/'+f'features_train_concated_{n}.pickle'
    targets_train_file = 'ex/'+f'targets_train_concated_{n}.pickle'
    features_test_file = 'ex/'+f'features_test_concated_{n}.pickle'
    targets_test_file = 'ex/'+f'targets_test_concated_{n}.pickle'
    name_model = f'models/model_v{version}_{n}_m{mode}_e{num_epochs}_concated.chkp'
    metrci_name =  f'models/model_v{version}_{n}_m{mode}_e{num_epochs}_concated.metric'


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

model = model_register()
model.gen_datasets(features_train, features_test, targets_train, targets_test, mode=mode)
if version == 'FNN':
    if mode == 'token':
        model.get_model(classifier_v1_token, vocab_size, mode='add')
    elif mode == 'bert':
        model.get_model(classifier_v1_bert, bertmodel, mode='add')
    else:
        model.get_model(classifier_v1)
elif version == 'T':
    if mode == 'token':
        model.get_model(classifier_v2_token, vocab_size, mode='add')
    elif mode == 'bert':
        model.get_model(classifier_v2_bert, bertmodel, mode='add')
    else:
        model.get_model(classifier_v2)
elif version == 'RES':
    model.get_model(classifier_v3)
elif version == 'RES_FNN':
    if n == 0:
        model.get_model(classifier_v1t3)
    elif n == 1:
        model.get_model(classifier_v1t3_token)
elif version == 'RES_RES':
    if n == 0:
        model.get_model(classifier_v3t3)
    elif n == 1:
        model.get_model(classifier_v3t3_token)
if os.path.exists(name_model) and load_model:
    checkpoint = torch.load(name_model)
    model.model.load_state_dict(checkpoint['model_state_dict'])

if train:
    model.train(num_epochs, testing=False)

model.test(to_print=True, epoch='-', count_zero=True)

with open(metrci_name, 'w') as f:
    f.write(model.text)

if not load_model:
    torch.save({
                'model_state_dict': model.model.state_dict(),
                }, name_model)

