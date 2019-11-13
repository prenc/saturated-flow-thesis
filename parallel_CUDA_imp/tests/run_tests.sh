#!/usr/bin/env bash
#:       Title: run_tests.sh - Runs performence tests for all .cu files in pwd.
#:    Synopsis: run_tests.sh
#:        Date: 2019-10-10
#:      Author: Paweł Renc
#:     Options: none
## Script metadata
scriptname=${0##*/}			# name that script is invoked with
## Constants
ca_size_arr=("100" "100" "1000" "1000")
block_size_arr=("16" "32" "16" "32")
iterations_arr=("1000" "1000" "1000" "1000")
params_file="params.h"
output_file_name="prof_summary"
## Func definitions
info () { #@ DESCRIPTION: print information about running process
          #@ USAGE: ok information
          #@ REQUIRES: scriptname
  printf "${scriptname}: [\e[34mINFO\e[0m] %s\n" "$1" >&2
}

generate_params() {
  match_ca_size="#define CA_SIZE "
  match_itarations="#define SIMULATION_ITERATIONS "
  match_block_size="#define BLOCK_SIZE "

  new_params_file="/tmp/new_params_ble_ble_ble"

  printf "" > "${new_params_file}"
  while read -r line; do
      if [[ $line =~ ^"${match_ca_size}" ]]; then
        line="${match_ca_size}${ca_size}"
      elif [[ $line =~ ^"${match_itarations}" ]]; then
        line="${match_itarations}${iterations}"
      elif [[ $line =~ ^"${match_block_size}" ]]; then
        line="${match_block_size}${block_size}"
      fi
      printf "%s\n" "${line}" >> "${new_params_file}"
  done < "${params_file}"

  cat ${new_params_file} > ${params_file}
  rm -f ${new_params_file}
}

compile_cuda_files () {
  files_to_compile=("${PWD}"/memory_*.cu)

  for file_name_path in "${files_to_compile[@]}"; do
    file_name=${file_name_path##*/}
    info "Compiling $file_name... (${ca_size}, ${iterations}, ${block_size})"
    nvcc "${file_name}" -o "${file_name%\.cu}"_compiled
  done

  files_to_test=("${PWD}"/memory_*_compiled)
}

profile_programs () {
  for file_name_path in "${files_to_test[@]}"; do
    file_name=${file_name_path##*/}
    info "Profiling ${file_name%_*}... (${ca_size}, ${iterations}, ${block_size})"
    sudo /usr/local/cuda/bin/nvprof --unified-memory-profiling off ./"${file_name}" 2> "${file_name%_compiled}"_profiling
  done

  profiling_data=("${PWD}"/memory_*_profiling)
}

parse_profile_outputs () {

  data=("memory_type" "ca_size" "iterations" "block_size" "total_time" "kernel_avg" "kernel_min"	"kernel_max")

  for prof in "${profiling_data[@]}"; do
    while read -r line; do
        if [[ $line =~ ^"GPU activities:" ]]; then
            values=(${line})
            break
        fi
    done < "${prof}"

    data+=("${prof##*/}" "$ca_size" "$iterations" "$block_size" "${values[3]}" "${values[5]}" "${values[6]}" "${values[7]}")
  done

  printf "%s;%s;%s;%s;%s;%s;%s;%s\n" "${data[@]}" >> "${output_file_name}"
}

## Script body
start=$(date +%s)
printf "" > "${output_file_name}"

for i in "${!ca_size_arr[@]}"; do
  ca_size=${ca_size_arr[i]}
  iterations=${iterations_arr[i]}
  block_size=${block_size_arr[i]}

  generate_params

  compile_cuda_files

  profile_programs

  parse_profile_outputs

done

end=$(date +%s)
info "Script time: $((end-start))"

exit 0