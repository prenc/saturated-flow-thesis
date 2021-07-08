#!/bin/bash

function displaytime {
	local T=$1
	local D=$((T/60/60/24))
	local H=$((T/60/60%24))
	local M=$((T/60%60))
	local S=$((T%60))
	(( $D > 0 )) && printf '%dd' $D
	(( $H > 0 )) && printf '%dh' $H
	(( $M > 0 )) && printf '%dm' $M
	printf '%ds\n' $S
}

TEST_REPETITIONS=10
elapsed_time=0
diff_time=0

compiled_dir="compiled"
rm -r compiled
./model_analyzer.py $1 -c

today=$(date +%d.%m)
base_dir=$([[ ! -z "$SUMMARIES_DIR" ]] && echo $SUMMARIES_DIR || echo .)
test_out_dir="$base_dir/$today/$1"
mkdir -p "$test_out_dir"

for rep in $(seq "${TEST_REPETITIONS}"); do
	for file in ${compiled_dir}/*; do
		echo Run test for target : $file
		file_name="$(basename -- "$file")"
		file_dir=${test_out_dir}/${file_name}
		mkdir -p ${file_dir}
		remaining="$(( $TEST_REPETITIONS - $rep ))"

		estimated_remaining=$(displaytime $(( $elapsed_time * $remaining / $rep )))

		echo -ne "Progress: ${rep}/${TEST_REPETITIONS} " \
			"elap./est.:$(displaytime ${elapsed_time})/${estimated_remaining}\r"

		start=$(date +%s)
		test_label=$(printf "%02d" "$rep")

		./${file}
		mv out/coverage_${file_name}.csv "$file_dir"/coverage_"${test_label}".csv

		diff_time="$(( $(date +%s) - $start ))"
		elapsed_time=$(( $elapsed_time + ${diff_time} ))
	done
done
echo
