# 与えられた文に対する応答を生成

import subprocess
import os
import sentencepiece as spm


spm_path = "dialogue/spm/twitter.model"
model_path = "dialogue/models/model_sp_step_100000.pt"

src_path = "tmp.txt"
output_path = "output/98_pred.txt"

# SentencePieceのモデルロード
sp = spm.SentencePieceProcessor()
sp.Load(spm_path)

sentence = input("input sentence: ")
if not sentence:
    sentence = "今日も一日お疲れさまでした！"   # テストデータより

# 分割してファイルに書き込み
pieces = sp.EncodeAsPieces(sentence)
with open(src_path, mode='w') as out_f:
    out_f.write(' '.join(pieces))

# 応答生成してファイルに書き込み
cmd_str = "python OpenNMT-py/translate.py "\
        f"-src {src_path} -model {model_path} -output {output_path} -replace_unk"
cmd = cmd_str.split()

subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# 応答読み込み
with open(output_path) as f:
    hyp_pieces = f.read()

# 分割をdecode
hyp_text = sp.DecodePieces(hyp_pieces.split())
print(hyp_text)

# inputのsentence書き込んだファイル消す
os.remove(src_path)


"""

result:
お疲れ様でした!

"""