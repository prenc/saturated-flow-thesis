#!/usr/bin/env bash

CA_SIZE=100
BLOCK_SIZE=16
INTERATION_NO=1000

files_to_compile=("${PWD}"/memory_*.cu)

for file_name_path in "${files_to_compile[@]}"; do
  file_name=${file_name_path##*/}
  echo "Compiling $file_name..."
  nvcc "${file_name}" -o "${file_name%\.cu}"_compiled
done

files_to_test=("${PWD}"/memory_*_compiled)

for file_name_path in "${files_to_test[@]}"; do
  file_name=${file_name_path##*/}
  echo "Profiling ${file_name%%_*}..."
  sudo nvprof ./"${file_name}" --unified-memory-profiling off 2> "${file_name%_compiled}"_profiling
done

profiling_data=("${PWD}"/memory_*_profiling)

data=("memory_type" "ca_size" "block_size" "iterations" "total_time" "kernel_avg" "kernel_min"	"kernel_max")

for prof in "${profiling_data[@]}"; do
  while read -r line; do
      if [[ $line =~ ^"GPU activities:" ]]; then
          values=(${line})
          break
      fi
  done < "${prof}"

  data+=("${prof##*/}" "$CA_SIZE" "$BLOCK_SIZE" "$INTERATION_NO" "${values[3]}" "${values[5]}" "${values[6]}" "${values[7]}")
done

output_file_name="prof_summary"
printf "%s;%s;%s;%s;%s;%s;%s;%s\n" "${data[@]}" > "$output_file_name"

exit 0