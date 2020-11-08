import logging
import os
import subprocess
from os.path import basename
from shutil import move

from ModelAnalyzer.settings import (
    COMPILED_DUMP,
    CMAKE_BUILD_DIR,
    ALWAYS_COMPILE,
)


class ProgramCompiler:
    def __init__(self, target_names):
        self._log = logging.getLogger(self.__class__.__name__)
        self._target_names = target_names

    def compile_test(self, test_spec):
        return self._find_executables(test_spec)

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
        exit_code = subprocess.run(["make", "-C", f"{CMAKE_BUILD_DIR}", "-j10", target]).returncode
        if exit_code == 0:
            move(os.path.join(CMAKE_BUILD_DIR, basename(target)), output_path)
        return exit_code