# Flaskを用いて対話サーバを構築

import subprocess
import os
import sentencepiece as spm
from flask import Flask, render_template, request

spm_path = "dialogue/spm/twitter.model"
model_path = "dialogue/models/sp_model_step_100000.pt"

src_path = "tmp_src.txt"
output_path = "tmp_out.txt"

app = Flask(__name__)

# SentencePieceのモデルロード
sp = spm.SentencePieceProcessor()
sp.Load(spm_path)


title = "チャット - SentencePiece"
dialogue_list = []


def get_reply(text):
    # 分割してファイルに書き込み
    pieces = sp.EncodeAsPieces(text)
    with open(src_path, mode='w') as out_f:
        out_f.write(' '.join(pieces))

    # 応答生成してファイルに書き込み
    cmd_str = "python OpenNMT-py/translate.py "\
            f"-src {src_path} -model {model_path} -output {output_path} -replace_unk -beam_size 10"
    cmd = cmd_str.split()
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 応答読み込み
    with open(output_path) as f:
        hyp_pieces = f.read()

    # 分割をdecode
    reply = sp.DecodePieces(hyp_pieces.split())

    # 書き込んだファイル消す
    os.remove(src_path)
    os.remove(output_path)

    return reply


@app.route("/", methods=["GET","POST"])
def chat():
    # 初期化
    if request.method == "GET":
        return render_template("chat.html", title=title, dialogue_list=[])
        
    # text受け取って応答生成
    input_text = request.form["input_text"]
    reply = get_reply(input_text)
    text_pair = (input_text, reply)
    dialogue_list.append(text_pair)
    return render_template("chat.html", title=title, dialogue_list=dialogue_list)


app.run(debug=True)
