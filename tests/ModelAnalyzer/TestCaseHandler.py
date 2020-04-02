import os
import time

from ModelAnalyzer.DefaultParamsKeeper import DefaultParamsKeeper
from ModelAnalyzer.settings import COMPILED_DUMP, PROFILING_DUMP, SUMMARIES_DUMP
from ModelAnalyzer.test_steps.ParamsGenerator import ParamsGenerator
from ModelAnalyzer.test_steps.ProgramCompilerAndRunner import (
    ProgramCompilerAndRunner,
)
from ModelAnalyzer.test_steps.ResultsHandler import ResultsHandler


class TestCaseHandler:
    def __init__(self, script_time):
        self._create_dirs()
        self.script_start_time = script_time

    @staticmethod
    def _create_dirs():
        dirs = [COMPILED_DUMP, PROFILING_DUMP, SUMMARIES_DUMP]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def perform_test_case(self, test_name, test_params):
        dpk = DefaultParamsKeeper()
        pg = ParamsGenerator(test_name)
        pcar = ProgramCompilerAndRunner(test_params["test_src"])
        rg = ResultsHandler(
            test_name,
            self.script_start_time,
            chart_params=test_params.get("chart_params", None),
        )
        dpk.create_params_copy()
        result_paths = []
        test_case_start_time = time.time()
        for test_spec in self._prepare_test_specs(test_params["test_specs"]):
            pg.generate(test_spec)
            intermediate_results_path = pcar.perform_test(test_spec)
            result_paths.extend(intermediate_results_path)
        test_case_elapsed_time = time.time() - test_case_start_time
        dpk.restore_params()
        return rg.save_results(result_paths, round(test_case_elapsed_time))

    @staticmethod
    def _prepare_test_specs(test_specs):
        return [
            {k: v for k, v in zip(test_specs.keys(), v)}
            for v in zip(*test_specs.values())
        ]
