#!/usr/bin/env bash
set -euo pipefail

echo "PATH=$PATH"
echo "which nsys: $(command -v nsys || true)"
echo "which -a nsys:"
which -a nsys || true
echo
for p in /usr/local/bin/nsys /usr/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys; do
  if [ -x "$p" ]; then
    echo "found executable: $p"
    "$p" --version || true
  fi
done
echo
nsys --version || true
echo
python - <<'PY'
import os
print('NSIGHT_SYSTEMS_VERSION=', os.environ.get('NSIGHT_SYSTEMS_VERSION'))
print('RAY_NSYS_BINARY=', os.environ.get('RAY_NSYS_BINARY'))
print('PATH=', os.environ.get('PATH'))
PY
echo
if [ -f "/lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/tools/inspect_ray_nsight_runtime.py" ]; then
  python3 /lustre/fsw/coreai_dlalgo_llm/users/pdalmia/nemo-rl-v0.5.0/tools/inspect_ray_nsight_runtime.py || true
fi
