help_slurm() {
    echo "📊 Monitor"
    echo "  sqgpu        → GPU jobs in gpu_short queue"
    echo "  sqjob ID     → all relevant informations about job"
    echo "  myprio       → your job priorities"
    echo "⚙️ Update"
    echo "  jobmem ID MB → change memory (MB)"
    echo "  jobtime ID T → change time limit"
    echo "  jobcpu ID N  → change CPU count"
}

# ---------- Monitor ----------

# Current gpu_short allocations
sqgpu() {
    squeue -p gpu,gpu_short -o "%.7i %.8u %.2t %.10M %.10l %.6D %.16R %.16b"
}

# All relevant info about job
sqjob() {
    squeue -j "$1" -o "%.10i %.8u %.10m %.2C %.10l %.16b %.8T %.16R %.16S"
}

# My job priorities / statuses
myprio() {
    sprio -u "$USER"
}

# ---------- Update ----------

# Change memory in MB
jobmem() {
    scontrol update JobId="$1" MinMemoryNode="$2"
}

# Change time limit
jobtime() {
    scontrol update JobId="$1" TimeLimit="$2"
}

# Change CPUs
jobcpu() {
    scontrol update JobId="$1" NumCPUs="$2"
}
