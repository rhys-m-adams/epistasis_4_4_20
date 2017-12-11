#!/bin/bash
# Simple SLURM sbatch example
#SBATCH --job-name=c3hs
#SBATCH --ntasks=1
#SBATCH --mem 40gb
#SBATCH --output=cdr1_out.txt
mpirun python local_epistatic_density1.py cdr1h $1

