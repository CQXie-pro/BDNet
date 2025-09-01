export CUDA_VISIBLE_DEVICES=0

model_name=BDNet

for seq_len in 720
do
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
  --label_len 48 \
  --pred_len $pred_len \
  --d_model 1024 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.002 \
  --batch_size 128 \
  --train_epochs 30 \
  --patience 3 \
  --moving_avg 25 \
  --top_k 5 \
  --model_type "mlp" \
  --multi_cycle 1
done
done
