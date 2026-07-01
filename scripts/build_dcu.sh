#!/bin/bash
export CUDA_HOME=/opt/dtk/cuda/cuda

xmake clean --all

xmake f -c --hygon-dcu=y --ccl=y --graph=y --cuda=$CUDA_HOME --aten=y --flash-attn-prebuilt=/usr/local/lib/python3.10/dist-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so

xmake build && xmake install

xmake build _infinicore && xmake install _infinicore

pip install -e . --no-build-isolation