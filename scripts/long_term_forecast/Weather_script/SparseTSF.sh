export CUDA_VISIBLE_DEVICES=0

model_name=SparseTSF

seq_len=720
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
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --learning_rate 0.02 \
  --batch_size 256 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 30 \
  --patience 5
done
