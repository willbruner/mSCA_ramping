#!/bin/bash

#SBATCH --account=p32032
#SBATCH --partition=short
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=10
#SBATCH --time=04:00:00
#SBATCH --mem=190g
#SBATCH --job-name=sweepModel


module purge all
module load python-miniconda3
eval "$(conda shell.bash hook)"
conda activate mp_opto

python --version
python bulkModel.py
