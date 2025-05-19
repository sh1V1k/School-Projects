#!/bin/bash
#SBATCH --job-name="Milestone1_CPU"
#SBATCH --output="Milestone1_CPU.out"
#SBATCH --error="Milestone1_CPU.err"
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

#srun ./m1_cpu 1000 > m1_cpu.out

# Generate running time information
 { time srun ./m1_cpu 1000 > m1_cpu.out; } 2> time.out

# Generate gprof output
#srun ./m1_cpu 1000 && gprof -Q ./m1_cpu gmon.out > outfile
