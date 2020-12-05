# データの準備

from sklearn.model_selection import train_test_split
import random

data_path = "dialogue/twitter_data/twitter_pair_example.txt"
result_path = "dialogue/twitter_data/orig/"

src_list = []
tgt_list = []

with open(data_path) as f:
    data_lines = f.read().split('\n')

for line in data_lines:
    text_pair = line.split('\t')
    # pairになってないのも混じってるので除去
    if len(text_pair) != 2:
        continue
    src_list.append(text_pair[0])
    tgt_list.append(text_pair[1])


train_src, tmp_src, train_tgt, tmp_tgt = train_test_split(src_list, tgt_list, train_size=0.99)
valid_src, test_src, valid_tgt, test_tgt = train_test_split(tmp_src, tmp_tgt, train_size=0.5)


def write_data(data_list, fname):
    with open(fname, mode='w') as out_f:
        for data in data_list:
            out_f.write(data+"\n")

write_data(train_src, result_path+"train_src.txt")
write_data(train_tgt, result_path+"train_tgt.txt")
write_data(valid_src, result_path+"valid_src.txt")
write_data(valid_tgt, result_path+"valid_tgt.txt")
write_data(test_src, result_path+"test_src.txt")
write_data(test_tgt, result_path+"test_tgt.txt")


# spm学習用
train_all = [*train_src, *train_tgt]
random.shuffle(train_all)
write_data(train_all, result_path+"train_all.txt")


"""

ペアになってなかったのは以下の8件

not pair: ['舘米沢店さん、ありがとうございます✨見て下さってるみなさんのおかげです✨）']
not pair: ['なかったことに気付けず、グッズの頒布を未然に防げなかったことについて申し訳ない気持ちでいっぱいです。']
not pair: ['もっと早くにツイートしなければと思いながら、考えをまとめられず、今日になってしまったことを始めに謝らせてください。遅くなって申し訳ありません。']
not pair: ['ぬいぐるみだけでなく、缶バッジの無配も止めるべきでした。同人イベントが初参加の彼女に同人活動がいかに繊細なものか、自分にできるアドバイスがあったのではないかと深く後悔しています。']
not pair: ['住所提供させてもらいました。  ⚠️元払いでお願いします⚠️  宛先']
not pair: ['〒501-6241']
not pair: ['岐阜県羽島市竹鼻町2499-17']
not pair: ['']

"""