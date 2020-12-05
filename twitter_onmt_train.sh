python OpenNMT-py/train.py \
 -train_steps 200000 \
 -data dialogue/preprocess/sp/twitter \
 -save_model dialogue/models/model_sp \
 -report_every 100 \
 -save_checkpoint_steps 100000 \
 -keep_checkpoint 3 \
 -gpu_ranks 0