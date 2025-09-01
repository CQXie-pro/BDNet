export CUDA_VISIBLE_DEVICES=0

model_name=CycleNet
seq_len=720
model_type='linear'

for pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --d_model 512 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --cycle 168 \
  --model_type $model_type \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 30 \
  --patience 3
done
