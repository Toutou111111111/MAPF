#!/usr/bin/env bash

if [ ! -d "./logs/MAPF/ETTm1" ]; then
  mkdir -p ./logs/MAPF/ETTm1
fi

model_name=MAPF
root_path_name=./data/ETT/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
seq_len=512
random_seed=2026

for pred_len in 96 192 336
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
    --trend_scales '97,193' \
    --wavelet_scales '2,4,8' \
    --wavelet_kernel_size 17 \
    --periods '96' \
    --seasonal_orders '3' \
    --fusion_gate 1 \
    --dropout 0.05 \
    --gpu 0 \
    --itr 1 \
    --batch_size 128 \
    --learning_rate 0.005 >logs/MAPF/ETTm1/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 720
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
    --trend_scales '97,193' \
    --wavelet_scales '2,4,8' \
    --wavelet_kernel_size 17 \
    --periods '96' \
    --seasonal_orders '3' \
    --fusion_gate 1 \
    --dropout 0.05 \
    --gpu 0 \
    --itr 1 \
    --batch_size 128 \
    --learning_rate 0.01 >logs/MAPF/ETTm1/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done