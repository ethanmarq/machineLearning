#!/bin/bash

#SBATCH --job-name train_UNETwRESNET
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8 
#SBATCH --gpus-per-node v100:1
#SBATCH --mem 12gb
#SBATCH --time 24:00:00


module load anaconda3
source activate torch


cd /home/marque6/Research\ 2024/ResNet\ CaT


python train.py