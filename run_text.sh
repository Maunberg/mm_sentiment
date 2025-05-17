#python train_parse.py --features_type 'text' --version 'FNN' --mode 'tfidf' --n 0
#python train_parse.py --features_type 'text' --version 'FNN' --mode 'bert' --n 0 --bertmodel 'Gherman/bert-base-NER-Russian'
#python train_parse.py --features_type 'text' --version 'FNN' --mode 'bert' --n 1 --bertmodel 'DeepPavlov/rubert-base-cased'
#python train_parse.py --features_type 'text' --version 'FNN' --mode 'bert' --n 2 --bertmodel 'cointegrated/rubert-tiny2-cedr-emotion-detection'
#python train_parse.py --features_type 'text' --version 'FNN' --mode 'token' --n 0

#python train_parse.py --features_type 'text' --version 'T' --mode 'tfidf' --n 0
#python train_parse.py --features_type 'text' --version 'T' --mode 'bert' --n 0 --bertmodel 'Gherman/bert-base-NER-Russian'
python train_parse.py --features_type 'text' --version 'T' --mode 'bert' --n 1 --bertmodel 'DeepPavlov/rubert-base-cased'
python train_parse.py --features_type 'text' --version 'T' --mode 'bert' --n 2 --bertmodel 'cointegrated/rubert-tiny2-cedr-emotion-detection'
python train_parse.py --features_type 'text' --version 'T' --mode 'token' --n 0


python train_parse.py --features_type 'text' --version 'RES' --mode 'tfidf' --n 0
python train_parse.py --features_type 'text' --version 'RES' --mode 'bert' --n 0 --bertmodel 'Gherman/bert-base-NER-Russian'
python train_parse.py --features_type 'text' --version 'RES' --mode 'bert' --n 1 --bertmodel 'DeepPavlov/rubert-base-cased'
python train_parse.py --features_type 'text' --version 'RES' --mode 'bert' --n 2 --bertmodel 'cointegrated/rubert-tiny2-cedr-emotion-detection'
python train_parse.py --features_type 'text' --version 'RES' --mode 'token' --n 0

python get_table.py
