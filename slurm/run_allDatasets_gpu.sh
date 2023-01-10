#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p qTRDGPUH
#SBATCH --mem=20gb
#SBATCH --gres=gpu:V100:1
#SBATCH -e log_files/error%A-%a.err
#SBATCH -o log_files/out%A-%a.out
#SBATCH -t 7200
#SBATCH -J BrainAge_10000
#SBATCH -A trends53c17
#SBATCH --oversubscribe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mduda@gsu.edu
#SBATCH --exclude=arctrdgn002

sleep 7s
export OMP_NUM_THREADS=1
##export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
#NODE=$(hostname)

echo $HOSTNAME >&2
module load python
source /userapp/virtualenv/mduda/venv/bin/activate

python allDatasets_example_test_gpu.py


sleep 7s
