export CUDA_VISIBLE_DEVICES=1

model_name=SparseTSF

seq_len=840
for pred_len in 168 336 504 672 840
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
  --pred_len $pred_len \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --learning_rate 0.03 \
  --batch_size 128 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 30 \
  --patience 5
done
