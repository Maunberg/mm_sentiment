import pickle

ex_dir = '/home/dutov@ad.speechpro.com/work/course/mm_sentiment/'

window = 4
overlap = 0
n = 0
concated_n = 1
audio = f'_w{window}_o{overlap}.pickle'
text = f'_{n}.pickle'
audio_ftype = 'mfcc'
text_ftype = 'token'


for add in [['features_train', 'targets_train'], ['features_test', 'targets_test']]:
    result = {'audio':[], 'text':[]}
    with open(ex_dir+'/ex/'+add[0]+audio, 'rb') as f:
        data_audio_f = pickle.load(f)[audio_ftype]
    with open(ex_dir+'/ex/'+add[1]+audio, 'rb') as f:
        data_audio_t = pickle.load(f)
    
    with open(ex_dir+'/ex/'+add[0]+text, 'rb') as f:
        data_text_f = pickle.load(f)[text_ftype]
    with open(ex_dir+'/ex/'+add[1]+text, 'rb') as f:
        data_text_t = pickle.load(f)
        
    for a, i in zip(data_audio_f, data_audio_t['id']):
        result['audio'].append(a)
        result['text'].append(data_text_f[int(i)])
    print(len(result['audio']))
    print(len(result['text']))
    print(len(data_audio_t['id']))
        
    with open(ex_dir+'/ex/'+add[0]+f'_concated_{concated_n}.pickle', 'wb') as f:
        pickle.dump(result, f)
    with open(ex_dir+'/ex/'+add[1]+f'_concated_{concated_n}.pickle', 'wb') as f:
        pickle.dump(data_audio_t, f)
    
        