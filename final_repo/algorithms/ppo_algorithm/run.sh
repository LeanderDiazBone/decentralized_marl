#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --account=3dv
#SBATCH --output=%j.out
#SBATCH --mincpus=8
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=30000MB

. /etc/profile.d/modules.sh
module add cuda/11.8

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID
#
# binary to execute
set -o errexit

srun python3 -u main.py
echo finished at: `date`
exit 0;