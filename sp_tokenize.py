# SentencePieceでtokenizeしたデータの作成

import sentencepiece as spm

orig_path = "dialogue/twitter_data/orig/"
tok_path = "dialogue/twitter_data/tok/sp/"
spm_prefix = "dialogue/spm/twitter"


spm.SentencePieceTrainer.Train(
    f"--input={orig_path}train_all.txt --model_prefix={spm_prefix} --vocab_size=8000")


# モデル読み込み
sp = spm.SentencePieceProcessor()
sp.Load(spm_prefix + ".model")


def tokenize_data(model, in_fname, out_fname):
    with open(in_fname) as in_f:
        lines = in_f.readlines()
    
    with open(out_fname, mode='w') as out_f:
        for line in lines:
            text = line.strip("\n")
            pieces = model.EncodeAsPieces(text)
            tokens = ' '.join(pieces)
            out_f.write(tokens + "\n")


tokenize_data(sp, orig_path+"train_src.txt", tok_path+"train_src.txt")
tokenize_data(sp, orig_path+"train_tgt.txt", tok_path+"train_tgt.txt")
tokenize_data(sp, orig_path+"valid_src.txt", tok_path+"valid_src.txt")
tokenize_data(sp, orig_path+"valid_tgt.txt", tok_path+"valid_tgt.txt")