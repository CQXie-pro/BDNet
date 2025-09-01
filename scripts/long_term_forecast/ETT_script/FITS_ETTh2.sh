export CUDA_VISIBLE_DEVICES=0

model_name=FITS

seq_len=720
for pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --learning_rate 0.0005 \
  --batch_size 64 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 50 \
  --patience 5 \
  --H_order 6
done
