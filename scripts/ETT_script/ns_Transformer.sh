python -u run.py \
  --is_training 1 \
  --root_path dataset/dataset/ETT-small \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model ns_Transformer \
  --data ETTh2 \
  --features MS \
  --seq_len 30 \
  --label_len 0 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --gpu 0 \
  --des 'Exp_h256_l2' \
  --p_hidden_dims 96 96 \
  --p_hidden_layers 2 \
  --itr 1 &

# python -u run.py \
#   --is_training 1 \
#   --root_path dataset/dataset/ETT-small \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_192 \
#   --model ns_Transformer \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --gpu 0 \
#   --des 'Exp_h64_l2' \
#   --p_hidden_dims 64 64 \
#   --p_hidden_layers 2 \
#   --itr 1  &

# python -u run.py \
#   --is_training 1 \
#   --root_path dataset/dataset/ETT-small \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_336 \
#   --model ns_Transformer \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --gpu 2 \
#   --des 'Exp_h256_l2' \
#   --p_hidden_dims 256 256 \
#   --p_hidden_layers 2 \
#   --itr 1  &

# python -u run.py \
#   --is_training 1 \
#   --root_path dataset/dataset/ETT-small \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_720 \
#   --model ns_Transformer \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --gpu 3 \
#   --des 'Exp_h256_l2' \
#   --p_hidden_dims 256 256 \
#   --p_hidden_layers 2 \
#   --itr 1  &