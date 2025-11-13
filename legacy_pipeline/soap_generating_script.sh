#!/bin/bash

#SBATCH -c 10

conda activate pyemma

python soap_generating_script.py