import os
import subprocess
import time
from shutil import rmtree

from ModelAnalyzer.DefaultParamsKeeper import DefaultParamsKeeper
from ModelAnalyzer.settings import (
    COMPILED_DUMP,
    PROFILING_DUMP,
    SUMMARIES_DUMP,
    CMAKE_BUILD_DIR,
    CMAKE_LISTS_PATH
)
from ModelAnalyzer.test_steps.ParamsGenerator import ParamsGenerator
from ModelAnalyzer.test_steps.ProgramCompiler import (
    ProgramCompiler,
)
from ModelAnalyzer.test_steps.ResultsHandler import ResultsHandler
from ModelAnalyzer.test_steps.ProgramRunner import ProgramRunner
from enum import Enum


class TestCaseHandler:
    class Mode(Enum):
        COMPILATION = 1
        CUSTOM_PROFILING = 2

    def __init__(self, script_time, mode=Mode.CUSTOM_PROFILING):
        self.mode = mode
        self._create_dirs(mode)
        self.script_start_time = script_time

    def _create_dirs(self, mode):
        dirs = [COMPILED_DUMP, CMAKE_BUILD_DIR]
        if mode == self.Mode.CUSTOM_PROFILING:
            dirs.extend([PROFILING_DUMP, SUMMARIES_DUMP])
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def perform_test_case(self, test_name, test_params):
        self._build_test()
        dpk = DefaultParamsKeeper()
        pg = ParamsGenerator(test_name)
        pc = ProgramCompiler(test_params["targets"])
        dpk.create_params_copy()

        result = ""
        if self.mode == self.Mode.COMPILATION:
            self._perform_test_compilation(pg, pc, test_params)
        else:
            result = self._perform_custom_profiling(pg, pc, test_name, test_params)

        dpk.restore_params()
        self._clean_build()
        return result

    @staticmethod
    def _build_test():
        subprocess.run(
            [
                "cmake",
                f"-B{CMAKE_BUILD_DIR}",
                f"-H{CMAKE_LISTS_PATH}"
            ]
        )

    @staticmethod
    def _clean_build():
        rmtree(CMAKE_BUILD_DIR, ignore_errors=True)

    def _perform_test_compilation(self, pg, pc, test_params):
        for test_spec in self._prepare_test_specs(test_params["params"]):
            pg.generate(test_spec)
            pc.compile_test(test_spec)

    def _perform_custom_profiling(self, pg, pc, test_name, test_params):
        pr = ProgramRunner(test_params["targets"])
        rg = ResultsHandler(
            test_name,
            self.script_start_time,
            chart_params=test_params.get("chart_params", None),
        )

        result_paths = []
        test_case_start_time = time.time()
        for test_spec in self._prepare_test_specs(test_params["params"]):
            pg.generate(test_spec)
            executables = pc.compile_test(test_spec)
            intermediate_results_path = pr.perform_test(test_spec, executables_data=executables)
            result_paths.extend(intermediate_results_path)
        test_case_elapsed_time = time.time() - test_case_start_time

        return rg.save_results(result_paths, round(test_case_elapsed_time))

    @staticmethod
    def _prepare_test_specs(test_specs):
        return [
            {k: v for k, v in zip(test_specs.keys(), v)}
            for v in zip(*test_specs.values())
        ]
