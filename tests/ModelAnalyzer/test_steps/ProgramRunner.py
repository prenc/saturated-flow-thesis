import json
import logging
import os
import subprocess
import time

from ModelAnalyzer.settings import (
    COMPILED_DUMP,
    PROFILING_DUMP,
    TIMES_EACH_PROGRAM_IS_RUN,
)


class ProgramRunner:
    def __init__(self, target_names):
        self._log = logging.getLogger(self.__class__.__name__)
        self._target_names = target_names

    def perform_test(self, test_spec, executables_data):
        return self._run_programs(test_spec, executables_data)

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
