#!/bin/bash
COLOR_ESC="\x1B["
COLOR_SUCC="32"
COLOR_FAIL="91"
COLOR_RESET="0"

logger() {
	local paint="$1"
	local message="$2"

	message="${COLOR_ESC}${paint}m${message} ${COLOR_ESC}${COLOR_RESET}m"
	echo -ne "$message"
}

logger_color(){
	local paint=$1
	shift 

	for message in "$@";
	do
		logger "$paint" "$message"
	done

	echo
}

succ(){
	logger_color $COLOR_SUCC $@
}

fail(){
	logger_color $COLOR_FAIL $@
}

script_dir="scripts"
test_script="compareMatrix.pl"
test_script=${script_dir}/${test_script}

output_dir="out"

./model_analyzer.py output

files=($output_dir/*)
total=${#files[@]}
comp_idx=1

total_diff=""

f1=${files[1]}
for f2 in ${files[@]}; do
	echo -ne "Comparing: $comp_idx/$total\n"
	comp_idx=$(( comp_idx + 1))
	output=$(perl ${test_script} $f1 $f2)
	if [[ "$output" = "OK" ]] || [[ "$f1" = "$f2" ]]
	then
		succ "OK"
	else
		diff="$f1 $f2 $output"
		fail "$diff"
		total_diff="$total_diff$diff \n"
	fi
done

if [[ "$total_diff" = "" ]]
then
	succ "All imlpementations are valid"
else
	fail "Invalid output: ${total_diff}"
fi

