#!/bin/sh
set -x #echo on
export PYTHONPATH=$PYTHONPATH:$1

command_file=`basename "$0"`
script_file=evaluate.py
gpu=0
data=/mnt/data/datasets/dcase2019/dev/task1a
model=c2d_resnet18_atresn_128_80
model_path=/mnt/data/ATReSN-Net/log_dcase_c2d_resnet18_atresn_128_80_8_sn8_train/model-104.ckpt
num_threads=6
num_segs=8
timebins=80
freqbins=128
num_classes=10
sn=8
fcn=5
dump_dir=log_dcase_c2d_resnet18_preact_sn_128_80_8_sn${sn}_eval_fcn5
log_file=$dump_dir.txt


python3 $script_file \
    --gpu $gpu \
    --data $data \
    --model $model \
    --model_path $model_path \
    --num_threads $num_threads \
    --num_segs $num_segs \
    --timebins $timebins \
    --freqbins $freqbins \
    --num_classes $num_classes \
    --sn $sn \
    --dump_dir $dump_dir \
    --fcn $fcn \
    --command_file $command_file \
    > $log_file 2>&1 &
