#!/usr/bin/env bash

if [ ! -d "./logs/MAPF/Weather" ]; then
  mkdir -p ./logs/MAPF/Weather
fi

model_name=MAPF
root_path_name=./data/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=weather
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
    --enc_in 21 \
    --trend_scales '145,289' \
    --wavelet_scales '1,2,4,8' \
    --wavelet_kernel_size 37 \
    --periods '144' \
    --seasonal_orders '3' \
    --seasonal_gate_bias 0.0 \
    --seasonal_scale_init 1.0 \
    --fusion_gate 1 \
    --dropout 0.05 \
    --gpu 0 \
    --itr 1 \
    --batch_size 128 \
    --learning_rate 0.001 >logs/MAPF/Weather/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 192
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
    --enc_in 21 \
    --trend_scales '145' \
    --wavelet_scales '1,2,4,8' \
    --wavelet_kernel_size 37 \
    --periods '144' \
    --seasonal_orders '3' \
    --seasonal_gate_bias 0.0 \
    --seasonal_scale_init 1.0 \
    --fusion_gate 1 \
    --dropout 0.05 \
    --gpu 0 \
    --itr 1 \
    --batch_size 128 \
    --learning_rate 0.0005 >logs/MAPF/Weather/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 336
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
    --enc_in 21 \
    --trend_scales '145' \
    --wavelet_scales '1,2,4,8' \
    --wavelet_kernel_size 17 \
    --periods '144' \
    --seasonal_orders '3' \
    --seasonal_gate_bias 0.0 \
    --seasonal_scale_init 1.0 \
    --fusion_gate 1 \
    --dropout 0.05 \
    --gpu 0 \
    --itr 1 \
    --batch_size 128 \
    --learning_rate 0.0005 >logs/MAPF/Weather/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
    --enc_in 21 \
    --trend_scales '145' \
    --wavelet_scales '2,4,8' \
    --wavelet_kernel_size 37 \
    --periods '144' \
    --seasonal_orders '3' \
    --fusion_gate 1 \
    --dropout 0.05 \
    --gpu 0 \
    --itr 1 \
    --batch_size 128 \
    --learning_rate 0.0005 >logs/MAPF/Weather/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
