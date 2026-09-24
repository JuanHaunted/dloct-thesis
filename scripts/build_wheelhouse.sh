#!/bin/bash
# Build an offline wheel bundle on your machine (with internet) for a cluster without it.
# Targets Linux x86_64, CPython 3.12 (Apolo's python/3.12 module), glibc 2.28 (Rocky 8).
#   bash scripts/build_wheelhouse.sh && rsync -avP wheelhouse <user>@apolo.eafit.edu.co:~/dloct-thesis/
set -euo pipefail
rm -rf wheelhouse && mkdir wheelhouse
uv export --frozen --no-dev --no-hashes --no-emit-project --format requirements-txt \
    | grep -v '^\s*#' | grep -v '^-e' > wheelhouse/requirements.txt
uvx pip download -r wheelhouse/requirements.txt -d wheelhouse \
    --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 \
    --python-version 3.12 --implementation cp --only-binary=:all: \
    --extra-index-url https://download.pytorch.org/whl/cu126
uvx pip download uv_build -d wheelhouse --only-binary=:all: --python-version 3.12 --platform manylinux_2_28_x86_64
du -sh wheelhouse
