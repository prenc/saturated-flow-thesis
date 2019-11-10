#!/usr/bin/env bash

files_to_compile=("${PWD}"/memory_*.cu)

for file_name_path in "${files_to_compile[@]}"; do
  file_name=${file_name_path##*/}
  docker run -v "${PWD}":/x nvidia/cuda:latest nvcc /x/"${file_name}" -o /x/"${file_name%\.cu}"_compiled
done

files_to_test=("${PWD}"/memory_*_compiled)