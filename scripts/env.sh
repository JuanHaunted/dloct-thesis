# Activate the project environment inside jobs and shells (sourced, not executed):
#   source scripts/env.sh
# Works for both setups made by scripts/setup_cluster.sh: a uv-managed or a module-python venv.
if [ -f .venv/.module-python ]; then
    module load "$(cat .venv/.module-python)" 2>/dev/null || true
fi
case $- in *u*) _dloct_nounset=1 ;; *) _dloct_nounset=0 ;; esac
set +u   # activate scripts reference unset variables
# shellcheck disable=SC1091
source .venv/bin/activate
[ "$_dloct_nounset" = 1 ] && set -u
unset _dloct_nounset
