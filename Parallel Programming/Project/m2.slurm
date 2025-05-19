#!/bin/bash
#SBATCH --job-name="Milestone2"
#SBATCH --output="Milestone2.out"
#SBATCH --error="Milestone2.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

srun ./m2 100 > m2.out

#{ time srun ./m2 10000 > m2.out; } 2> time.out

#perf,nvperf

# Generate nsys profile information, output will be in profile.out
# srun nsys profile --stats=true ./m2 > profile.out
# srun ncu --set full -f -o analysis_file ./m2 10000
