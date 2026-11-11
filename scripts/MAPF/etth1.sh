#!/usr/bin/env bash

if [ ! -d "./logs/MAPF/ETTh1" ]; then
  mkdir -p ./logs/MAPF/ETTh1
fi

model_name=MAPF
root_path_name=./data/ETT/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
seq_len=512
random_seed=2026

for pred_len in 96
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
    --enc_in 7 \
    --trend_scales '49,97,169' \
    --wavelet_scales '2,4,6' \
    --wavelet_kernel_size 19 \
    --periods '24,168' \
    --seasonal_orders '8,1' \
    --fusion_gate 1 \
    --dropout 0.05 \
    --gpu 0 \
    --itr 1 \
    --batch_size 128 \
    --learning_rate 0.0005 >logs/MAPF/ETTh1/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 192 336 720
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
    --enc_in 7 \
    --trend_scales '49,97,169' \
    --wavelet_scales '2,3,4' \
    --wavelet_kernel_size 19 \
    --periods '24,168' \
    --seasonal_orders '8,1' \
    --fusion_gate 1 \
    --dropout 0.05 \
    --gpu 0 \
    --itr 1 \
    --batch_size 128 \
    --learning_rate 0.01 >logs/MAPF/ETTh1/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done