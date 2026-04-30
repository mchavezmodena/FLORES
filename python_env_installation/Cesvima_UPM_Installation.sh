#!/usr/bin/env bash
set -e

### ------------------------------------------------------------
### 0. Project setup
### ------------------------------------------------------------

PROJECT_DIR="$(pwd)"
VENV_DIR="$PROJECT_DIR/flores_env"
PETSC_SRC="$PROJECT_DIR/petsc"
PETSC_ARCH="arch-linux-c-opt"
SLEPC_SRC="$PROJECT_DIR/slepc"

# --- GCC runtime necesario para libgfortran.so.4 (al final para no interferir) ---
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/media/apps/avx512/software/GCCcore/7.2.0/lib64"

### ------------------------------------------------------------
### WARNING: comprobacion de acceso a internet
### ------------------------------------------------------------

echo ">>> Checking internet access..."
if ! curl -sf --max-time 10 https://github.com -o /dev/null 2>&1; then
    echo ""
    echo "**********************************************************************"
    echo "  WARNING: No internet access detected from this node."
    echo ""
    echo "  This script needs to download PETSc dependencies (CMake, METIS,"
    echo "  ParMETIS) from the internet during the configure step."
    echo ""
    echo "  If you are running from a COMPUTE NODE, downloads may fail."
    echo "  To fix this, predownload the required files from the LOGIN NODE:"
    echo ""
    echo "    mkdir -p $PROJECT_DIR/downloads"
    echo "    wget https://github.com/Kitware/CMake/releases/download/v4.3.1/cmake-4.3.1.tar.gz \\"
    echo "         -O $PROJECT_DIR/downloads/cmake-4.3.1.tar.gz"
    echo "    git clone https://bitbucket.org/petsc/pkg-metis.git \\"
    echo "         $PROJECT_DIR/downloads/pkg-metis"
    echo "    git clone https://bitbucket.org/petsc/pkg-parmetis.git \\"
    echo "         $PROJECT_DIR/downloads/pkg-parmetis"
    echo ""
    echo "  Then update the --download-cmake, --download-metis and"
    echo "  --download-parmetis flags in this script to point to those"
    echo "  local paths and relaunch the job."
    echo ""
    echo "  Aborting installation."
    echo "**********************************************************************"
    echo ""
    exit 1
fi
echo ">>> Internet access OK"

### ------------------------------------------------------------
### WARNING: comprobacion de variables de modulos
### ------------------------------------------------------------

echo ">>> Checking required module variables..."

# Inicializar sistema de modulos si no esta disponible
if ! command -v module &>/dev/null; then
    if [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -f /usr/share/lmod/lmod/init/bash ]; then
        source /usr/share/lmod/lmod/init/bash
    elif [ -n "$MODULESHOME" ]; then
        source "$MODULESHOME/init/bash"
    fi
fi

# Carga automatica de modulos si no estan cargados
if [ -z "$EBROOTOPENMPI" ] || [ -z "$EBROOTOPENBLAS" ] || \
   [ -z "$EBROOTSCALAPACK" ] || [ -z "$EBROOTMUMPS" ]; then
    echo ">>> Module variables not set, attempting to load modules..."
    if command -v module &>/dev/null; then
        module purge
        module load foss/2021a
        module load MUMPS/5.4.0-foss-2021a-metis
        echo ">>> Modules loaded"
    fi
fi

# Verificacion final
MISSING_VARS=0
for VAR in EBROOTOPENMPI EBROOTOPENBLAS EBROOTSCALAPACK EBROOTMUMPS; do
    if [ -z "${!VAR}" ]; then
        echo "  ERROR: \$$VAR is not set"
        MISSING_VARS=1
    else
        echo "  OK: $VAR = ${!VAR}"
    fi
done

if [ "$MISSING_VARS" -eq 1 ]; then
    echo ""
    echo "**********************************************************************"
    echo "  ERROR: One or more required module variables are not set."
    echo ""
    echo "  Make sure the following modules are loaded before running this script"
    echo "  or that the script can load them automatically:"
    echo ""
    echo "    module purge"
    echo "    module load foss/2021a"
    echo "    module load MUMPS/5.4.0-foss-2021a-metis"
    echo ""
    echo "  Aborting installation."
    echo "**********************************************************************"
    exit 1
fi
echo ">>> All module variables OK"

### ------------------------------------------------------------
### 0.1 Virtual environment
### ------------------------------------------------------------

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo ">>> Creating virtual environment"
    python3 -m venv "$VENV_DIR"
else
    echo ">>> [SKIP] Virtual environment already exists"
fi

source "$VENV_DIR/bin/activate"

if ! python3 -c "import numpy" &>/dev/null; then
    echo ">>> Installing base Python packages"
    pip install --upgrade pip setuptools wheel numpy
else
    echo ">>> [SKIP] Base Python packages already installed"
fi

### ------------------------------------------------------------
### 1. PETSc (inplace, sin make install)
### ------------------------------------------------------------

# 1a. Clone
if [ ! -d "$PETSC_SRC/.git" ]; then
    echo ">>> Cloning PETSc"
    git clone -b release https://gitlab.com/petsc/petsc.git "$PETSC_SRC"
else
    echo ">>> [SKIP] PETSc source already cloned"
fi

export PETSC_DIR="$PETSC_SRC"
export PETSC_ARCH="arch-linux-c-opt"
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH"
export MPICC="$(which mpicc)"

# 1b. Configure
PETSC_CONFIGURE_DONE="$PETSC_SRC/.configure_done"

if [ -f "$PETSC_CONFIGURE_DONE" ]; then
    echo ">>> [SKIP] PETSc already configured"
else
    echo ">>> Configuring PETSc"
    cd "$PETSC_SRC"
    ./configure \
      --with-debugging=0 \
      --with-shared-libraries=1 \
      --with-mpi=1 \
      --with-scalar-type=complex \
      --with-mpi-dir="$EBROOTOPENMPI" \
      --with-blas-lapack-dir="$EBROOTOPENBLAS" \
      --with-scalapack-dir="$EBROOTSCALAPACK" \
      --with-mumps-dir="$EBROOTMUMPS" \
      --download-cmake \
      --download-metis \
      --download-parmetis
    touch "$PETSC_CONFIGURE_DONE"
    echo ">>> PETSc configure complete"
fi

# 1c. Build (sin make install)
if ls "$PETSC_SRC/$PETSC_ARCH/lib/libpetsc.so."* 1>/dev/null 2>&1; then
    echo ">>> [SKIP] PETSc already built"
else
    echo ">>> Building PETSc"
    cd "$PETSC_SRC"
    make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" all
fi

### ------------------------------------------------------------
### 2. SLEPc (inplace, sin make install)
### ------------------------------------------------------------

# 2a. Clone
if [ ! -d "$SLEPC_SRC/.git" ]; then
    echo ">>> Cloning SLEPc"
    git clone -b release https://gitlab.com/slepc/slepc.git "$SLEPC_SRC"
else
    echo ">>> [SKIP] SLEPc source already cloned"
fi

export SLEPC_DIR="$SLEPC_SRC"
export LD_LIBRARY_PATH="$SLEPC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH"

# 2b. Configure
SLEPC_CONFIGURE_DONE="$SLEPC_SRC/.configure_done"

if [ -f "$SLEPC_CONFIGURE_DONE" ]; then
    echo ">>> [SKIP] SLEPc already configured"
else
    echo ">>> Configuring SLEPc"
    cd "$SLEPC_SRC"
    ./configure
    touch "$SLEPC_CONFIGURE_DONE"
    echo ">>> SLEPc configure complete"
fi

# 2c. Build (sin make install)
if ls "$SLEPC_SRC/$PETSC_ARCH/lib/libslepc.so."* 1>/dev/null 2>&1; then
    echo ">>> [SKIP] SLEPc already built"
else
    echo ">>> Building SLEPc"
    cd "$SLEPC_SRC"
    make SLEPC_DIR="$SLEPC_DIR" PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" all
fi

### ------------------------------------------------------------
### 3. mpi4py
### ------------------------------------------------------------

if ! python3 -c "import mpi4py" &>/dev/null; then
    echo ">>> Installing mpi4py"
    pip uninstall -y mpi4py || true
    pip cache purge
    pip install --no-binary=mpi4py mpi4py
else
    echo ">>> [SKIP] mpi4py already installed"
fi

### ------------------------------------------------------------
### 4. petsc4py
### ------------------------------------------------------------

if ! python3 -c "import petsc4py" &>/dev/null; then
    echo ">>> Installing petsc4py"
    pip uninstall -y petsc petsc4py || true
    pip cache purge
    CFLAGS="-O0 -g0" MAX_JOBS=1 pip install \
      --no-binary=petsc4py \
      --no-binary=petsc \
      --no-deps \
      "petsc4py==3.25.0"
else
    echo ">>> [SKIP] petsc4py already installed"
fi

### ------------------------------------------------------------
### 5. slepc4py
### ------------------------------------------------------------

if ! python3 -c "import slepc4py" &>/dev/null; then
    echo ">>> Installing slepc4py"
    pip uninstall -y slepc slepc4py || true
    pip cache purge
    CFLAGS="-O0 -g0" MAX_JOBS=1 pip install \
      --no-binary=slepc4py \
      --no-binary=slepc \
      --no-deps \
      "slepc4py==3.25.0"
else
    echo ">>> [SKIP] slepc4py already installed"
fi

### ------------------------------------------------------------
### 6. Patch activate (solo si no está ya parcheado)
### ------------------------------------------------------------

ACTIVATE_FILE="$VENV_DIR/bin/activate"

if ! grep -q "PETSC_DIR" "$ACTIVATE_FILE"; then
    echo ">>> Patching activate script"
    cat << EOF >> "$ACTIVATE_FILE"

# --- Modulos del cluster ---
if command -v module &>/dev/null; then
    module load foss/2021a
    module load MUMPS/5.4.0-foss-2021a-metis
fi

# --- GCC runtime (libgfortran.so.4) ---
export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:/media/apps/avx512/software/GCCcore/7.2.0/lib64"

# --- PETSc configuration (inplace) ---
export PETSC_DIR="$PETSC_SRC"
export PETSC_ARCH="arch-linux-c-opt"

# --- SLEPc configuration (inplace) ---
export SLEPC_DIR="$SLEPC_SRC"

# --- Runtime library paths ---
export LD_LIBRARY_PATH="\$PETSC_DIR/\$PETSC_ARCH/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="\$SLEPC_DIR/\$PETSC_ARCH/lib:\$LD_LIBRARY_PATH"

# --- MPI compiler wrapper ---
export MPICC="\$PETSC_DIR/\$PETSC_ARCH/bin/mpicc"
EOF
else
    echo ">>> [SKIP] activate script already patched"
fi

### ------------------------------------------------------------
### 7. Extra modules
### ------------------------------------------------------------

if ! python3 -c "import scipy" &>/dev/null; then
    echo ">>> Installing extra Python packages"
    pip install netCDF4 scipy matplotlib
else
    echo ">>> [SKIP] Extra Python packages already installed"
fi

### ------------------------------------------------------------
### 8. Sanity check
### ------------------------------------------------------------

echo ">>> Running sanity test"
python3 - << 'EOF'
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI

print("PETSc version:", PETSc.Sys.getVersion())
print("SLEPc version:", SLEPc.Sys.getVersion())
print("PETSc COMM_WORLD:", PETSc.COMM_WORLD.tompi4py().py2f())
print("mpi4py COMM_WORLD:", MPI.COMM_WORLD.py2f())
EOF

echo ">>> Installation complete!"