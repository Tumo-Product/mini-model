#!/bin/sh
# Owner-only: copy the two trained checkpoints out of the Modal dev volume they were trained on.
# Teammates without access to that Modal workspace get weights/ from the owner instead (about 90 MB).
set -e
cd "$(dirname "$0")/.."
VOL="${JIGAI_VOLUME:-irish-music-model-dev}"
mkdir -p weights/tsALL-s6000-v1 weights/d482-s2400
modal volume get --force "$VOL" runs/tsALL-s6000-v1/model.pt weights/tsALL-s6000-v1/model.pt
modal volume get --force "$VOL" runs/2026-09-14T03-37-13Z/model.pt weights/d482-s2400/model.pt
echo "done: weights/*/model.pt (config.json with each vocabulary is already in the repo)"
