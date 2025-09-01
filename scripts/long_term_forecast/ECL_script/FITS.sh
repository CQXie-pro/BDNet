export CUDA_VISIBLE_DEVICES=1

model_name=FITS

seq_len=720
for pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --learning_rate 0.0005 \
  --batch_size 64 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 50 \
  --patience 5
done
