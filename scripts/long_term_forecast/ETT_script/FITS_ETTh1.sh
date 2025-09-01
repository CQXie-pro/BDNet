export CUDA_VISIBLE_DEVICES=1

model_name=FITS

seq_len=720
for pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --batch_size 64 \
  --itr 1 \
  --train_epochs 100 \
  --patience 20 \
  --H_order 6
done
