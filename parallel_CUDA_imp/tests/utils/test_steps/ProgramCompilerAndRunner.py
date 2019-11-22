import json
import logging
import os
import subprocess
import time
from shutil import move

from utils.common.TimeCounter import TimeCounter


class ProgramCompilerAndRunner:
    COMPILED_DIR_PATH = "compiled"
    PROFILING_DIR_PATH = "profiling"
    RESULTS_DIR_PATH = "results"

    def __init__(self, file_names):
        self._log = logging.getLogger(self.__class__.__name__)

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

    def perform_test(self, test_spec):
        executables = self._find_executables(test_spec)
        return self._run_programs(test_spec, executables)

    def _find_executables(self, test_spec):
        executable_files = []
        for root, name in self._program_paths:
            new_file_name = f"{name.split('.')[0]}_" + "_".join(
                [str(value) for value in test_spec.values()]
            )
            output_path = os.path.join(self.COMPILED_DIR_PATH, new_file_name)
            exit_code = 0
            if os.path.isfile(output_path):
                self._log.info(f"Found '{new_file_name}'. No need to compile.")
            else:
                exit_code = self._compile_file(root, name, output_path)
            if exit_code == 0:
                executable_files.append((name, new_file_name))
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
        self._log.info(f"Compiling using 'make': '{name}'.")
        subprocess.run(["cmake", f"-B{root}", f"-H{root}"])
        exit_code = subprocess.run(["make", "-C", root])
        if exit_code == 0:
            move(os.path.join(root, name.split(".")[0]), output_path)

    def _compile_cu_file(self, root, name, output_path):
        self._log.info(f"Compiling using 'nvcc': '{name}'.")
        return subprocess.run(
            ["nvcc", os.path.join(root, name), "-o", output_path]
        ).returncode

    def _run_programs(self, test_spec, executable_names):
        results = []
        for src_name, executable_name in executable_names:
            self._log.info(f"Testing '{executable_name}'.")
            tc = TimeCounter()
            tc.start()
            subprocess.run([f"./{self.COMPILED_DIR_PATH}/{executable_name}"])
            tc.stop()
            results.append(
                self._save_test_results(
                    src_name,executable_name, test_spec, tc.elapsed_time
                )
            )
        return results

    def _save_test_results(
        self, src_name, executable, test_spec, elapsed_time, exit_code=0
    ):
        results = {
            "src_name": src_name,
            "executable": executable,
            "datastamp": time.time(),
            "elapsed_time": elapsed_time,
            "exit_code": exit_code,
            **test_spec,
        }
        result_path = os.path.join(self.PROFILING_DIR_PATH, executable)
        with open(result_path, "w") as result_file:
            json.dump(results, result_file, indent=4)
            return result_path
