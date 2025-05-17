import os
dir = '/home/dutov@ad.speechpro.com/work/course/mm_sentiment/models/'
files = [i for i in os.listdir(dir) if i.count('.metric')]
data = {}
voice_files = [i for i in files if i.count('_voice.metric')]
text_files = [i for i in files if i.count('_text.metric')]
image_files = [i for i in files if i.count('_image.metric')]
mm_files = [i for i in files if i.count('_concated.metric')]

def get_res(res, files, dir=dir):
    res = {'EMO_F1':[], 'EMO_RMSE':[], 'SENT_F1':[], 'SENT_RMSE':[]}
    for file in files:
        splitted = file.replace('.metric', '').split('_')[1:][:-1]
        for i in splitted:
            if i[0] not in res:
                res[i[0]] = []
            res[i[0]].append(i[1:])
        with open(dir+'/'+file) as f:
            text = f.readlines()
        res['EMO_F1'].append([i for i in text if i.count('F1 mean')][0].split(': ')[-1].strip())
        res['SENT_F1'].append([i for i in text if i.count('F1 Sentiment')][0].split(': ')[-1].strip())
        res['EMO_RMSE'].append([i for i in text if i.count('RMSE mean')][0].split(': ')[-1].strip())
        res['SENT_RMSE'].append([i for i in text if i.count('RMSE Sentiment')][0].split(': ')[-1].strip())
    return res

def get_text_up(res):
    text = ''
    column_width = max(len(key) for key in res.keys()) + 2
    header = " | ".join(f"{key:<{column_width}}" for key in res.keys())
    text = header + '\n' + "-" * (len(header)) + '\n'
    for values in zip(*res.values()):
        row = " | ".join(f"{value:<{column_width}}" for value in values)
        text += row + '\n'
    return text

def get_text_left(res):
    text = ''
    column_width = max(len(str(item)) for row in res.values() for item in row) + 2
    header_width = max(len(key) for key in res.keys()) + 2
    text += f"{'':<{header_width}} | " + " | ".join(f"{i:<{column_width}}" for i in range(len(next(iter(res.values()))))) + '\n'
    text += "-" * (header_width + 3 + (column_width + 3) * len(next(iter(res.values())))) + '\n'
    for key, values in res.items():
        text += f"{key:<{header_width}} | " + " | ".join(f"{str(value):<{column_width}}" for value in values) + '\n'
    return text

with open(dir+'/full_audio.txt', 'w') as f:
    if voice_files != []:
        r = get_res({}, voice_files)
        text = get_text_left(r)
        f.write(text)

with open(dir+'/full_text.txt', 'w') as f:
    if text_files != []:
        r = get_res({}, text_files)
        text = get_text_left(r)
        f.write(text)

with open(dir+'/full_image.txt', 'w') as f:
    if image_files != []:
        r = get_res({}, image_files)
        text = get_text_left(r)
        f.write(text)


with open(dir+'/full_mm.txt', 'w') as f:
    if mm_files != []:
        r = get_res({}, mm_files)
        text = get_text_left(r)
        f.write(text)