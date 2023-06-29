#!/bin/bash

#SBATCH --job-name 'test'
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time 1-0
#SBATCH --partition batch_ce_ugrad
#SBATCH -o logs/slurm-%A-%x.out

if test -d model;then
    echo "model dir exist"
else
    mkdir model
fi

if test -f model/sam_vit_h_4b8939.pth;then
    echo "Segment-anythin model exist"
else
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P model/
fi

python train.py

exit 0