#!/bin/sh
set -x #echo on
export PYTHONPATH=$PYTHONPATH:$1

command_file=`basename "$0"`
script_file=train.py
gpu=0
data=/mnt/data/datasets/dcase2019/dev/task1a
model=c2d_resnet18_atresn_128_80
model_path=pretrained_models/ResNet18-DCASE19.npz
batch_size=12
learning_rate=0.002
num_threads=6
num_segs=8
timebins=80
freqbins=128
num_classes=10
sn=10
decay_step=40
mixup=False
mixup_alpha=0.4
log_dir=log_dcase_${model}_${num_segs}_sn${sn}_augpos_train
log_file=$log_dir.txt


python $script_file \
    --decay_step $decay_step \
    --gpu $gpu \
    --data $data \
    --model $model \
    --model_path $model_path \
    --log_dir $log_dir \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --num_threads $num_threads \
    --num_segs $num_segs \
    --timebins $timebins \
    --freqbins $freqbins \
    --num_classes $num_classes \
    --sn $sn \
    --decay_step $decay_step \
    --mixup $mixup \
    --mixup_alpha $mixup_alpha \
    --command_file $command_file \
    > $log_file 2>&1 &
