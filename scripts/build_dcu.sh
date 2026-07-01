#!/bin/bash
set -euo pipefail
export CUDA_HOME=/opt/dtk/cuda/cuda
export XMAKE_ROOT=y

xmake clean --all

xmake f -y -c --hygon-dcu=y --ccl=y --graph=y --cuda=$CUDA_HOME --aten=y --flash-attn-prebuilt=/usr/local/lib/python3.10/dist-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so

xmake build -y && xmake install -y

xmake build -y _infinicore && xmake install -y _infinicore

pip install -e . --no-build-isolation
