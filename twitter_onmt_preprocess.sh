path="dialogue/twitter_data/tok/sp"
#path="dialogue/twitter_data/tok/mecab"
python OpenNMT-py/preprocess.py \
 -train_src $path/train_src.txt \
 -train_tgt $path/train_tgt.txt \
 -valid_src $path/valid_src.txt \
 -valid_tgt $path/valid_tgt.txt \
 -save_data dialogue/preprocess/sp/twitter \
 -overwrite