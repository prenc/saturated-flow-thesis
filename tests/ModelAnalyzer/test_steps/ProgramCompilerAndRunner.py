import json
import logging
import os
import subprocess
import time
from os.path import basename
from shutil import move

from ModelAnalyzer.settings import (
    COMPILED_DUMP,
    PROFILING_DUMP,
    TIMES_EACH_PROGRAM_IS_RUN,
    CMAKE_BUILD_DIR,
    ALWAYS_COMPILE,
)


class ProgramCompilerAndRunner:
    def __init__(self, target_names):
        self._log = logging.getLogger(self.__class__.__name__)
        self._target_names = target_names

    def perform_test(self, test_spec):
        executables_data = self._find_executables(test_spec)
        return self._run_programs(test_spec, executables_data)

    def _find_executables(self, test_spec):
        executables_data = []
        for target in self._target_names:
            new_file_name = f"{target.split('.')[0]}_" + "_".join(
                [
                    str(value)
                    for value in test_spec.values()
                    if isinstance(value, int)
                ]
            )
            output_path = os.path.join(COMPILED_DUMP, new_file_name)
            exit_code = 0
            if os.path.isfile(output_path) and not ALWAYS_COMPILE:
                self._log.info(f"Found '{new_file_name}'. No need to compile.")
            else:
                exit_code = self._compile_target(target, output_path)
            executables_data.append(
                {
                    "src_name": target,
                    "executable_name": new_file_name,
                    "compiling_exit_code": exit_code,
                }
            )
        return executables_data

    def _compile_target(self, target, output_path):
        self._log.info(f"Compiling using 'make': '{target}'.")
        exit_code = subprocess.run(["make", "-C", f"{CMAKE_BUILD_DIR}", target]).returncode
        if exit_code == 0:
            move(os.path.join(CMAKE_BUILD_DIR, basename(target)), output_path)
        return exit_code

    def _run_programs(self, test_spec, executables_data):
        results_paths = []
        for executable_data in executables_data:
            if executable_data["compiling_exit_code"] == 0:
                result_path = self._run_program(
                    {**executable_data, **test_spec}
                )
                results_paths.append(result_path)
        return results_paths

    def _run_program(self, executable_data):
        self._log.info(f"Testing '{executable_data['executable_name']}'.")
        runs_times = []
        exit_code = 0
        for i in range(TIMES_EACH_PROGRAM_IS_RUN):
            run_start_time = time.time()
            exit_code = subprocess.run(
                [f"{COMPILED_DUMP}/{executable_data['executable_name']}"]
            ).returncode
            run_elapsed_time = time.time() - run_start_time
            self._log.info(
                f"Test {i + 1}/{TIMES_EACH_PROGRAM_IS_RUN} has "
                f"been run in {run_elapsed_time // 60:.0f}m"
                f"{run_elapsed_time % 60:.0f}s"
            )
            runs_times.append(run_elapsed_time)
        return self._save_test_summary(
            executable_data, min(runs_times), exit_code
        )

    def _save_test_summary(self, executable_data, elapsed_time, exit_code):
        results_object = {
            **executable_data,
            "datastamp": round(time.time()),
            "elapsed_time": elapsed_time,
            "run_exit_code": exit_code,
        }
        result_file_path = os.path.join(
            PROFILING_DUMP, executable_data["executable_name"]
        )
        self._log.debug(f"Intermediate result save to '{result_file_path}'")
        with open(result_file_path, "w") as result_file:
            json.dump(results_object, result_file, indent=4)
            return result_file_path
