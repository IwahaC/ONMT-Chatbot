# 与えられた文に対する応答を生成 (MeCabバージョン)

import MeCab
import subprocess
import os

src_path = "tmp.mecab"
model_path = "dialogue/models/mecabmodel_step_100000.pt"
output_path = "output/98_pred.txt"

mecab_parser = MeCab.Tagger('-Owakati')


sentence = input("input sentence: ")
if not sentence:
    sentence = "今日も一日お疲れさまでした！"

words = mecab_parser.parse(sentence)

with open(src_path, mode='w') as out_f:
    out_f.write(words)


cmd_str = "python OpenNMT-py/translate.py "\
        f"-src {src_path} -model {model_path} -output {output_path} -replace_unk"
cmd = cmd_str.split()

subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

with open(output_path) as f:
    words = f.read().strip('\n')

# 空白除去
text = words.replace(' ', '')
print(text)

os.remove(src_path)


"""

result:
お疲れ様でした!

"""