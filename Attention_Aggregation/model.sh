#!/bin/sh


#SBATCH -p nvidia
#SBATCH --gres=gpu:1

python model.py
