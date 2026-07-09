#!/bin/bash

#SBATCH --account=p32032
#SBATCH --partition=short
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=40
#SBATCH --time=04:00:00
#SBATCH --mem=180g
#SBATCH --job-name=sweepRegsSW


module purge all
module load python-miniconda3
eval "$(conda shell.bash hook)"
conda activate mp_opto

python --version
python to_3_06052024_sweepRegs.py
