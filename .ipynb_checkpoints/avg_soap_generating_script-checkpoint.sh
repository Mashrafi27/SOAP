#!/bin/bash

#SBATCH -c 10

conda activate pyemma

python avg_soap_generating_script.py