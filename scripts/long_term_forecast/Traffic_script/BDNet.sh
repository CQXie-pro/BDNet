export CUDA_VISIBLE_DEVICES=1

model_name=BDNet

for seq_len in 720
do
for pred_len in 192 336 720
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
  --d_model 1024 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.002 \
  --batch_size 128 \
  --train_epochs 30 \
  --patience 3 \
  --moving_avg 25 \
  --top_k 5 \
  --model_type "mlp"
done
done
