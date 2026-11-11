#!/usr/bin/env bash

if [ ! -d "./logs/MAPF/traffic" ]; then
  mkdir -p ./logs/MAPF/traffic
fi

model_name=MAPF
root_path_name=./data/traffic/
data_path_name=traffic.csv
model_id_name=traffic
data_name=traffic
seq_len=512
random_seed=2026

for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=0 python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 862 \
    --trend_scales '49,97,169' \
    --wavelet_scales '1,2,4,8' \
    --wavelet_kernel_size 13 \
    --periods '24,168' \
    --seasonal_orders '8,1' \
    --fusion_gate 1 \
    --dropout 0.05 \
    --gpu 0 \
    --itr 1 \
    --batch_size 16 \
    --num_workers 0 \
    --learning_rate 0.005 >logs/MAPF/traffic/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
