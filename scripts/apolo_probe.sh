#!/bin/bash
# Collect the cluster facts the training setup depends on into ~/apolo_report.txt.
# Read-only: it installs nothing and changes nothing. Run on the Apolo login node:
#   bash apolo_probe.sh
# The last section starts a 5-minute GPU job; it may wait in the queue (max 15 min here).
OUT=${1:-$HOME/apolo_report.txt}
exec > >(tee "$OUT") 2>&1

section() { printf '\n===== %s =====\n' "$1"; }
net() {  # print HTTP status for a URL, or FAIL
    local code
    code=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' "$1" 2>/dev/null) || code=FAIL
    printf '  %-45s %s\n' "$1" "$code"
}

section "identity"
echo "user=$USER host=$(hostname) date=$(date -Is)"
head -2 /etc/os-release; ldd --version 2>&1 | head -1
echo "shell=$SHELL groups=$(id -Gn)"

section "partitions (sinfo)"
sinfo -s
sinfo -o "%P %a %l %D %c %m %G %N" 2>&1

section "accel-2 details"
scontrol show partition accel-2 2>&1 | grep -oE "(MaxTime|DefaultTime|MaxNodes|State|AllowAccounts|AllowGroups|AllowQos|QoS|TRES)=[^ ]*"
scontrol show node "$(sinfo -h -p accel-2 -o %N | head -1)" 2>&1 | grep -oE "(Gres|CPUTot|RealMemory|State|CfgTRES)=[^ ]*"

section "my account / QOS"
sacctmgr -nP show assoc user="$USER" format=cluster,account,partition,qos,maxwall 2>&1 | head -20
sacctmgr -nP show qos format=name,maxwall,maxtrespu,maxjobspu 2>&1 | head -20

section "storage"
df -h "$HOME" 2>&1
for d in /scratch /scratch-local /scratch-global /data /work /storage "/scratch/$USER" "/scratch-local/$USER"; do
    [ -e "$d" ] && { ls -ld "$d"; df -h "$d" | tail -1; }
done
quota -s 2>/dev/null || echo "quota: n/a"
command -v du-home >/dev/null && du-home 2>&1 | tail -5

section "internet from login node"
for u in https://pypi.org/simple/ https://download.pytorch.org/whl/cu126/ https://astral.sh/uv/install.sh \
         https://github.com https://www.googleapis.com; do net "$u"; done
env | grep -i _proxy || echo "  no proxy variables set"

section "tools and modules"
for t in rsync git curl wget python3 conda uv tmux screen; do printf '  %-8s %s\n' "$t" "$(command -v $t || echo -)"; done
module -t avail python 2>&1 | head -30
module -t avail cuda 2>&1 | head -15

section "GPU node (5-minute job on accel-2, waits up to 15 min in queue)"
timeout 900 srun -p accel-2 --gres=gpu:1 -N 1 -n 1 -c 2 --mem=4G -t 0-00:05:00 bash -c '
    echo "node=$(hostname) cpus=$(nproc)"; free -g | head -2
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
    nvidia-smi | head -5
    echo "internet from compute node:"
    for u in https://pypi.org/simple/ https://download.pytorch.org/whl/cu126/; do
        printf "  %-45s %s\n" "$u" "$(curl -sS -m 10 -o /dev/null -w %{http_code} "$u" 2>/dev/null || echo FAIL)"
    done
    df -h /tmp | tail -1
' 2>&1 || echo "GPU job did not run (queued too long, no permission for accel-2, or error above)"

section "done"
echo "report written to $OUT"
