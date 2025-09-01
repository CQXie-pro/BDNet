export CUDA_VISIBLE_DEVICES=0

model_name=BDNet
seq_len=720

for seq_len in 720
do
for pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --freq 10min \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --d_model 128 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --batch_size 128 \
  --train_epochs 50 \
  --patience 3 \
  --moving_avg 145 \
  --top_k 5 \
  --model_type "mlp"
done
done
