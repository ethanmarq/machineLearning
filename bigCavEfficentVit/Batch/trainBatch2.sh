#!/bin/bash

#SBATCH --job-name train_efficientVit
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8 
#SBATCH --gpus-per-node a100:1
#SBATCH --mem 32gb
#SBATCH --time 12:00:00


module load anaconda3
source activate torch


cd /home/marque6/Research\ 2024/EfficentVit\ on\ CaT/bigCav_EfficientVit


python train_seg_model.py configs/seg/cav/b0-lr3-n2-wd1.yaml --path runtest --weight_url ckpts/b0_modified.pt