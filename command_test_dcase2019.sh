#!/bin/sh
set -x #echo on
export PYTHONPATH=$PYTHONPATH:$1

command_file=`basename "$0"`
script_file=test.py
gpu=0
data=/mnt/data/datasets/dcase2019/dev/gulp_1_12_1024_480
model=c2d_resnet34_cp_112
model_path=/mnt/data/cpnet-master/log_dcase_c2d_resnet34_cp_112_12_1_train/bak_models/model-67.ckpt
num_threads=6
num_frames=12
frame_step=1
width=112
height=112
num_classes=10
full_size=112
fcn=6
dump_dir=log_dcase_c2d_resnet34_cp_112_12_1_train_test_6
log_file=$dump_dir.txt


python3 $script_file \
    --gpu $gpu \
    --data $data \
    --model $model \
    --model_path $model_path \
    --num_threads $num_threads \
    --num_frames $num_frames \
    --frame_step $frame_step \
    --width $width \
    --height $height \
    --num_classes $num_classes \
    --dump_dir $dump_dir \
    --fcn $fcn \
    --full_size $full_size \
    --command_file $command_file \
    > $log_file 2>&1 &
