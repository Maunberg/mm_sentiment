python train_parse.py --features_type 'voice' --version 'FNN' --window 1 --overlap 0 --mode 'spectr' --train 'False'
python train_parse.py --features_type 'voice' --version 'FNN' --window 1 --overlap 0.5 --mode 'spectr' --train 'False'
python train_parse.py --features_type 'voice' --version 'FNN' --window 2 --overlap 0 --mode 'spectr' --train 'False'
python train_parse.py --features_type 'voice' --version 'FNN' --window 2 --overlap 0.5 --mode 'spectr' --train 'False'
python train_parse.py --features_type 'voice' --version 'FNN' --window 3 --overlap 0 --mode 'spectr' --train 'False'
python train_parse.py --features_type 'voice' --version 'FNN' --window 4 --overlap 0 --mode 'spectr' --train 'False'

echo '\n\n\n\n\nMEL\n\n\n\n'

python train_parse.py --features_type 'voice' --version 'FNN' --window 1 --overlap 0 --mode 'mel'  --train 'False'
python train_parse.py --features_type 'voice' --version 'FNN' --window 1 --overlap 0.5 --mode 'mel' --train 'False'
python train_parse.py --features_type 'voice' --version 'FNN' --window 2 --overlap 0 --mode 'mel' --train 'False'
python train_parse.py --features_type 'voice' --version 'FNN' --window 2 --overlap 0.5 --mode 'mel' --train 'False'
python train_parse.py --features_type 'voice' --version 'FNN' --window 3 --overlap 0 --mode 'mel' --train 'False'
python train_parse.py --features_type 'voice' --version 'FNN' --window 4 --overlap 0 --mode 'mel' --train 'False'

echo '\n\n\n\n\nMFCC\n\n\n\n'

python train_parse.py --features_type 'voice' --version 'FNN' --window 1 --overlap 0 --mode 'mfcc' --train 'False'
python train_parse.py --features_type 'voice' --version 'FNN' --window 1 --overlap 0.5 --mode 'mfcc' --train 'False'
python train_parse.py --features_type 'voice' --version 'FNN' --window 2 --overlap 0 --mode 'mfcc' --train 'False'
python train_parse.py --features_type 'voice' --version 'FNN' --window 2 --overlap 0.5 --mode 'mfcc' --train 'False'
python train_parse.py --features_type 'voice' --version 'FNN' --window 3 --overlap 0 --mode 'mfcc' --train 'False'
python train_parse.py --features_type 'voice' --version 'FNN' --window 4 --overlap 0 --mode 'mfcc' --train 'False'

echo '\n\n\n\n\nSPECTR\n\n\n\n'

python train_parse.py --features_type 'voice' --version 'T' --window 1 --overlap 0 --mode 'spectr' --train 'False'
python train_parse.py --features_type 'voice' --version 'T' --window 1 --overlap 0.5 --mode 'spectr' --train 'False'
python train_parse.py --features_type 'voice' --version 'T' --window 2 --overlap 0 --mode 'spectr' --train 'False'
python train_parse.py --features_type 'voice' --version 'T' --window 2 --overlap 0.5 --mode 'spectr' --train 'False'
python train_parse.py --features_type 'voice' --version 'T' --window 3 --overlap 0 --mode 'spectr' --train 'False'
python train_parse.py --features_type 'voice' --version 'T' --window 4 --overlap 0 --mode 'spectr' --train 'False'

echo '\n\n\n\n\nMEL T\n\n\n\n'

python train_parse.py --features_type 'voice' --version 'T' --window 1 --overlap 0 --mode 'mel' --train 'False'
python train_parse.py --features_type 'voice' --version 'T' --window 1 --overlap 0.5 --mode 'mel' --train 'False'
python train_parse.py --features_type 'voice' --version 'T' --window 2 --overlap 0 --mode 'mel' --train 'False'
python train_parse.py --features_type 'voice' --version 'T' --window 2 --overlap 0.5 --mode 'mel' --train 'False'
python train_parse.py --features_type 'voice' --version 'T' --window 3 --overlap 0 --mode 'mel' --train 'False'
python train_parse.py --features_type 'voice' --version 'T' --window 4 --overlap 0 --mode 'mel' --train 'False'

echo '\n\n\n\n\nMFCC T\n\n\n\n'

python train_parse.py --features_type 'voice' --version 'T' --window 1 --overlap 0 --mode 'mfcc' --train 'False'
python train_parse.py --features_type 'voice' --version 'T' --window 1 --overlap 0.5 --mode 'mfcc' --train 'False'
python train_parse.py --features_type 'voice' --version 'T' --window 2 --overlap 0 --mode 'mfcc' --train 'False'
python train_parse.py --features_type 'voice' --version 'T' --window 2 --overlap 0.5 --mode 'mfcc' --train 'False'
python train_parse.py --features_type 'voice' --version 'T' --window 3 --overlap 0 --mode 'mfcc' --train 'False'
python train_parse.py --features_type 'voice' --version 'T' --window 4 --overlap 0 --mode 'mfcc' --train 'False'

python get_table.py

