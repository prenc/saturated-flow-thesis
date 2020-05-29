#!/usr/bin/env bash
nvcc river_unified.cu -o river_exe
./river_exe
./river_visualization.py