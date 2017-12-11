#!/bin/bash
# Simple SLURM sbatch example
#SBATCH --job-name=c3hs
#SBATCH --ntasks=1
#SBATCH --mem 40gb
#SBATCH --output=cdr3_out.txt
mpirun python local_epistatic_density3.py cdr3h $1

