#!/bin/bash
##------------------------ Start job description -----------------------

#SBATCH --partition=standard
#SBATCH --job-name=plot_eigf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=1:00:00
#SBATCH -e LOG/err-%j.log
#SBATCH -o LOG/out-%j.log

##------------------------ End job description ------------------------

# OpenMP / threading
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_SCHEDULE=guided
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Si usas MKL/OpenBLAS/NumExpr (muy común en Python), limita también:
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

./load_env_STAB_tool.sh

python3 plot_pval_eigfunction.py --dir eig_results/ --mode 1 64 