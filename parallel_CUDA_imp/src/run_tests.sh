#!/usr/bin/env bash
#:       Title: run_tests.sh - Runs performence tests for all .cu files in pwd.
#:    Synopsis: run_tests.sh
#:        Date: 2019-10-10
#:      Author: Paweł Renc
#:     Options: none
shopt -s nullglob 			# allow globs to return null string
## Script metadata
scriptname=${0##*/} # name that script is invoked with
## Constants
ca_size_arr=("100" "100" "1000" "1000")
block_size_arr=("16" "32" "16" "32")
iterations_arr=("1000" "1000" "10000" "10000")
params_file="params.h"
output_file_name="prof_summary"
compiled_dir="compiled"
profiling_dir="profiling"
## Func definitions
info() { #@ DESCRIPTION: print information about running process
  #@ USAGE: ok information
  #@ REQUIRES: scriptname
  printf "${scriptname}: [\e[34mINFO\e[0m] %s\n" "$1" >&2
}

generate_params() {
  match_ca_size="#define CA_SIZE "
  match_itarations="#define SIMULATION_ITERATIONS "
  match_block_size="#define BLOCK_SIZE "

  new_params_file="/tmp/new_params_ble_ble_ble"

  printf "" >"${new_params_file}"
  while read -r line; do
    if [[ $line =~ ^"${match_ca_size}" ]]; then
      line="${match_ca_size}${ca_size}"
    elif [[ $line =~ ^"${match_itarations}" ]]; then
      line="${match_itarations}${iterations}"
    elif [[ $line =~ ^"${match_block_size}" ]]; then
      line="${match_block_size}${block_size}"
    fi
    printf "%s\n" "${line}" >>"${new_params_file}"
  done <"${params_file}"

  cat ${new_params_file} >${params_file}
  rm -f ${new_params_file}

  file_name_attachement=_"${ca_size}"_"${iterations}"_"${block_size}"
}

compile_cuda_files() {

  [[ -d "${compiled_dir}" ]] || mkdir "${compiled_dir}"
  files_to_compile=("${PWD}"/*.cu)

  for file_name_path in "${files_to_compile[@]}"; do
    file_name_in=${file_name_path##*/}
    file_name_out=${file_name_in%\.cu}${file_name_attachement}

    if [[ ! -f "${compiled_dir}/${file_name_out}" ]]; then
      info "Compiling ${file_name_in}... (${ca_size}, ${iterations}, ${block_size})"
      nvcc "${file_name_in}" -o "${compiled_dir}/${file_name_out}"
    else
      info "Found ${file_name_out}. No need to compile again..."
    fi
  done
}

profile_programs() {

  [[ -d "${profiling_dir}" ]] || mkdir "${profiling_dir}"

  files_to_profile=("${PWD}"/"${compiled_dir}"/*)

  for file_name_path in "${files_to_profile[@]}"; do
    file_name=${file_name_path##*/}
    info "Profiling ${file_name}... (${ca_size}, ${iterations}, ${block_size})"
    command time -f "%E" ./"${compiled_dir}/${file_name}" 2>"${profiling_dir}/${file_name}"

  done
}

parse_profile_outputs() {
  profiling_data=("${PWD}"/"${profiling_dir}"/*)

  data=("memory_type" "ca_size" "iterations" "block_size" "total_time")

  for prof in "${profiling_data[@]}"; do
#    while read -r line; do
#      if [[ $line =~ ^"GPU activities:" ]]; then
#        read -r -a values <<<"${line}"
#        break
#      fi
#    done <"${prof}"
    line=$(head -n 1 "${prof}")

    data+=("${prof##*/}" "$ca_size" "$iterations" "$block_size" "${line}")
  done

  printf "%s;%s;%s;%s;%s\n" "${data[@]}" >>"${output_file_name}"
}

## Script body
start=$(date +%s)
printf "" >"${output_file_name}"

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
info "Script time: $((end - start))s"

exit 0

