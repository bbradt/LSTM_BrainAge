#!/bin/bash
#SBATCH -p qTRDGPUL,qTRDGPUM,qTRDGPUH
#SBATCH --mem=20gb
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -A trends53c17
#SBATCH --oversubscribe
echo ARGS "${@:1}"
eval "$(conda shell.bash hook)"
conda activate catalyst3.9
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/users2/bbaker/miniconda3/lib python main.py "${@:1}"

