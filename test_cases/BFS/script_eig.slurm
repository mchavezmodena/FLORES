#!/bin/bash
##------------------------ Start job description -----------------------
#SBATCH --partition=debug # standard
#SBATCH --job-name=eig
#SBATCH --ntasks=2
##SBATCH --ntasks-per-node=2      # 1 rango MPI por nodo
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=5      # todos los cores del nodo para OpenMP
#SBATCH --mem=80G
#SBATCH --time=0:30:00 # debug: 0:20:00
#SBATCH -e LOG/err-%j.log
#SBATCH -o LOG/out-%j.log
##------------------------ End job description ------------------------

# ── Threading ─────────────────────────────────────────────────────────────────
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}  
export OMP_SCHEDULE=guided
export OMP_PROC_BIND=close
export OMP_PLACES=cores


export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1                    # desactivado
export MKL_NUM_THREADS=1                         # desactivado
export NUMEXPR_NUM_THREADS=1                     # desactivado

# ── Paths ─────────────────────────────────────────────────────────────────────
SOLVER_PATH=/home/w059/w059589/apps/FLORES/solver
INI_FILE=BFS_eig.floresparam

# ── Environment ───────────────────────────────────────────────────────────────
source load_env_STAB_tool_ompi.sh

# ── Create LOG directory if it does not exist ─────────────────────────────────
mkdir -p LOG

# ── Run ───────────────────────────────────────────────────────────────────────
echo "========================================"
echo " Job ID       : ${SLURM_JOB_ID}"
echo " Job name     : ${SLURM_JOB_NAME}"
echo " Partition    : ${SLURM_JOB_PARTITION}"
echo " Num. tasks   : ${SLURM_NTASKS}"
echo " CPUs/task    : ${SLURM_CPUS_PER_TASK}"
echo " OMP threads  : ${OMP_NUM_THREADS}"
echo " Node list    : ${SLURM_NODELIST}"
echo " INI file     : ${INI_FILE}"
echo "========================================"

# mpirun -n ${SLURM_NTASKS} python3 ${SOLVER_PATH}/eig_simple.py ${INI_FILE}
mpirun -n ${SLURM_NTASKS}  python3 ${SOLVER_PATH}/eig_simple.py ${INI_FILE}