#!/usr/bin/env bash

if [ ! -d "./logs/MAPF/electricity" ]; then
  mkdir -p ./logs/MAPF/electricity
fi

model_name=MAPF
root_path_name=./data/electricity/
data_path_name=electricity.csv
model_id_name=electricity
data_name=electricity
seq_len=512
random_seed=2026

for pred_len in 96 192
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
    --enc_in 321 \
    --trend_scales '49,169' \
    --trend_router_hidden 32 \
    --trend_residual_detach 0 \
    --wavelet_scales '1,2,4,6' \
    --wavelet_kernel_size 19 \
    --wavelet_hidden 64 \
    --wavelet_threshold_init 0.01 \
    --wavelet_output_scale_init 1.0 \
    --periods '24,168' \
    --seasonal_orders '12,2' \
    --seasonal_ridge 0.01 \
    --seasonal_gate_hidden 16 \
    --seasonal_gate_bias 0.0 \
    --seasonal_scale_init 0.3 \
    --fusion_gate 1 \
    --fusion_gate_hidden 32 \
    --horizon_gate 1 \
    --horizon_gate_dim 8 \
    --dropout 0.05 \
    --gpu 0 \
    --itr 1 \
    --batch_size 16 \
    --num_workers 0 \
    --learning_rate 0.005 >logs/MAPF/electricity/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 336 720
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
    --enc_in 321 \
    --trend_scales '49,169' \
    --trend_router_hidden 32 \
    --trend_residual_detach 0 \
    --wavelet_scales '1,2,4,6' \
    --wavelet_kernel_size 19 \
    --wavelet_hidden 64 \
    --wavelet_threshold_init 0.01 \
    --wavelet_output_scale_init 1.0 \
    --periods '24,168' \
    --seasonal_orders '12,2' \
    --seasonal_ridge 0.01 \
    --seasonal_gate_hidden 16 \
    --seasonal_gate_bias 0.0 \
    --seasonal_scale_init 0.5 \
    --fusion_gate 1 \
    --fusion_gate_hidden 32 \
    --horizon_gate 1 \
    --horizon_gate_dim 8 \
    --dropout 0.05 \
    --gpu 0 \
    --itr 1 \
    --batch_size 16 \
    --num_workers 0 \
    --learning_rate 0.005 >logs/MAPF/electricity/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done