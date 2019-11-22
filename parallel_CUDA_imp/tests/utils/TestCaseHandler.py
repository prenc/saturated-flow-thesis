import os
import time

from utils.common.TimeCounter import TimeCounter
from utils.common.constants import (
    COMPILED_DIR_PATH,
    PROFILING_DIR_PATH,
    RESULTS_DIR_PATH,
)
from utils.test_steps.ParamsGenerator import ParamsGenerator
from utils.test_steps.ProgramCompilerAndRunner import ProgramCompilerAndRunner
from utils.test_steps.ResultsHandler import ResultsHandler


class TestCaseHandler:
    def __init__(self):
        self._create_dirs()
        self.script_start_time = int(time.time())

    @staticmethod
    def _create_dirs():
        dirs = [COMPILED_DIR_PATH, PROFILING_DIR_PATH, RESULTS_DIR_PATH]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def perform_test_case(self, test_name, test_params):
        pg = ParamsGenerator(test_name)
        pcar = ProgramCompilerAndRunner(test_params["test_src"])
        rg = ResultsHandler(test_name, self.script_start_time)
        test_case_counter = TimeCounter()

        result_paths = []
        test_case_counter.start()
        for test_spec in self._prepare_test_specs(test_params["test_specs"]):
            pg.generate(test_spec)
            run_tests_results = pcar.perform_test(test_spec)
            result_paths.extend(run_tests_results)
        test_case_counter.stop()
        rg.save_results(result_paths, test_case_counter.elapsed_time)

    @staticmethod
    def _prepare_test_specs(test_specs):
        return [
            {
                "block_size": block_size,
                "ca_size": ca_size,
                "iterations": iterations,
            }
            for block_size, ca_size, iterations in zip(
                test_specs["block_size"],
                test_specs["ca_size"],
                test_specs["iterations"],
            )
        ]
