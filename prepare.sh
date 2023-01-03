#!/usr/bin/env bash
conda activate plb
export PYTHONPATH=$PWD:$PWD/taichi_three:$PWD/PointFlow:$PWD/setvae:$PYTHONPATH
export PATH=$PWD/taichi_three:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH