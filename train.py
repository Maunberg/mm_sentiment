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

from utils import model_register
from utils import classifier_v2, classifier_v1

ex_dir = '/home/dutov@ad.speechpro.com/work/course/mm_sentiment/'
window = 2
version = 'T' #FNN #T
overlap = 0
overlap_test = 0
mode = 'mfcc' #mfcc #mel #spectr
train = False
load_model = True
num_epochs = 50


features_train_file = 'ex/'+f'features_train_w{window}_o{overlap}.pickle'
targets_train_file = 'ex/'+f'targets_train_w{window}_o{overlap}.pickle'
features_test_file = 'ex/'+f'features_test_w{window}_o{overlap_test}.pickle'
targets_test_file = 'ex/'+f'targets_test_w{window}_o{overlap_test}.pickle'
name_model = f'models/model_v{version}_m{mode}_w{window}_o{overlap}_e{num_epochs}.chkp'
metrci_name =  f'models/model_v{version}_m{mode}_w{window}_o{overlap_test}_e{num_epochs}.metric'

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
    model.get_model(classifier_v1)
elif version == 'T':
    model.get_model(classifier_v2)
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
