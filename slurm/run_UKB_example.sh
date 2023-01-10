#!/bin/bash
#SBATCH -p qTRDHM
#SBATCH --mem=20gb
#SBATCH -e log_files/error%A-%a.err
#SBATCH -o log_files/out%A-%a.out
#SBATCH -t 500
#SBATCH -J BrainAge_UKB2000
#SBATCH -A trends53c17
#SBATCH --oversubscribe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mduda@gsu.edu

sleep 7s
export OMP_NUM_THREADS=1
##export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
#NODE=$(hostname)

echo $HOSTNAME >&2
module load python
source /userapp/virtualenv/mduda/venv/bin/activate

python UKB_example.py


sleep 7s
