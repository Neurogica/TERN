#!/usr/bin/env bash
# Clone the third-party code used for the baselines (shallow, not vendored) and copy the Cola-GNN datasets.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/ext_repos" "$ROOT/data/colagnn"
cd "$ROOT/ext_repos"
[ -d tslib ]   || git clone --depth 1 https://github.com/thuml/Time-Series-Library.git tslib
[ -d colagnn ] || git clone --depth 1 https://github.com/amy-deng/colagnn.git colagnn
[ -d epignn ]  || git clone --depth 1 https://github.com/Xiefeng69/EpiGNN.git epignn
cp -n colagnn/data/*.txt "$ROOT/data/colagnn/" 2>/dev/null || true
echo "datasets: $(ls "$ROOT/data/colagnn" | tr '\n' ' ')"
