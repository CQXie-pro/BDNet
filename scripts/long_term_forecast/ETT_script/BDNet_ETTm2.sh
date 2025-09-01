export CUDA_VISIBLE_DEVICES=1

model_name=BDNet
seq_len=720

for top_k in 1 3 7 9 15
do
for pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --freq 15min \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --d_model 512 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --batch_size 128 \
  --train_epochs 50 \
  --patience 3 \
  --moving_avg 97 \
  --top_k $top_k
done
done