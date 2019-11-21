import json
import logging
import os
import subprocess
from shutil import move

from utils.common.TimeCounter import TimeCounter


class ProgramTester:
    COMPILED_DIR_PATH = "compiled"
    PROFILING_DIR_PATH = "profiling"
    RESULTS_DIR_PATH = "results"

    def __init__(self, file_names, file_suffix):
        self._log = logging.getLogger(self.__class__.__name__)

        self._file_suffix = file_suffix
        self._program_paths = self._find_paths(file_names)
        self._create_dirs()

    @staticmethod
    def _find_paths(file_names):
        found_paths = []
        for root, dirs, files in os.walk("../.."):
            for file in files:
                if file in file_names:
                    found_paths.append((root, file))
        return found_paths

    def _create_dirs(self):
        dirs = [
            self.COMPILED_DIR_PATH,
            self.PROFILING_DIR_PATH,
            self.RESULTS_DIR_PATH,
        ]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def perform_test(self):
        executables = self._find_executables()
        self._run_programs(executables)

    def _find_executables(self):
        executable_files = []
        for root, name in self._program_paths:
            new_file_name = name.split(".")[0] + self._file_suffix
            output_path = os.path.join(self.COMPILED_DIR_PATH, new_file_name)
            exit_code = 0
            if os.path.isfile(output_path):
                self._log.info(f"Found '{new_file_name}'. No need to compile.")
            else:
                exit_code = self._compile_file(root, name, output_path)
            if exit_code == 0:
                executable_files.append(new_file_name)
        return executable_files

    def _compile_file(self, root, name, output_path):
        if name.endswith(".c"):
            return self._compile_c_file(root, name, output_path)
        elif name.endswith(".cu"):
            return self._compile_cu_file(root, name, output_path)
        else:
            self._log.warning(f"Could not compile. Unknown extension: {name}")
            return -1

    def _compile_c_file(self, root, name, output_path):
        self._log.info(f"Compiling using 'make': {name}")
        subprocess.run(["cmake", f"-B{root}", f"-H{root}"])
        exit_code = subprocess.run(["make", "-C", root])
        if exit_code == 0:
            move(os.path.join(root, name.split(".")[0]), output_path)

    def _compile_cu_file(self, root, name, output_path):
        self._log.info(f"Compiling using 'nvcc': {name}")
        return subprocess.run(
            ["nvcc", os.path.join(root, name), "-o", output_path]
        ).returncode

    def _run_programs(self, file_names):
        for file_name in file_names:
            self._log.info(f"Testing '{file_name}'.")
            tc = TimeCounter()
            tc.start()
            subprocess.run([f"./{self.COMPILED_DIR_PATH}/{file_name}"])
            tc.stop()
            self._save_test_results(file_name, tc.elapsed_time)

    def _save_test_results(self, file_name, elapsed_time, exit_code=0):
        results = {"elapsed_time": elapsed_time, "exit_code": exit_code}
        with open(
            os.path.join(self.PROFILING_DIR_PATH, file_name), "w"
        ) as result_file:
            json.dump(results, result_file, indent=4)
