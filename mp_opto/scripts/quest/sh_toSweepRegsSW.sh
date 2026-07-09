#!/bin/bash

#SBATCH --account=p32032
#SBATCH --partition=short
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=20
#SBATCH --time=04:00:00
#SBATCH --mem=180g
#SBATCH --job-name=to_SW


module purge all
module load python-miniconda3
eval "$(conda shell.bash hook)"
conda activate mp_opto

python --version
python toSweepRegsSW.py
