import json
import logging
import os
from collections import defaultdict

from utils.settings import RESULTS_DIR_PATH


class ResultsHandler:
    def __init__(self, test_name, script_time, chart_params=None):
        self._log = logging.getLogger(self.__class__.__name__)
        self._script_time = int(script_time)
        self._test_name = test_name
        self._chart_params = chart_params

    def save_results(self, run_programs, test_case_time):
        data = self._gather_results(run_programs, test_case_time)
        self._log.info(
            f"Test case '{self._test_name}' took {test_case_time // 60:.0f}m"
            f"{test_case_time % 60:.0f}s"
        )
        return self._save_summary_in_file(data)

    def _gather_results(self, run_programs_paths, test_time):
        summary_results = {
            "test_name": self._test_name,
            "test_time": test_time,
            "run_tests": defaultdict(list),
            "chart_params": self._chart_params,
        }
        for path in run_programs_paths:
            with open(path, "r") as result_file:
                result_json = json.load(result_file)
                src_name = result_json["src_name"].split(".")[0]
                summary_results["run_tests"][src_name].append(result_json)
        return summary_results

    def _save_summary_in_file(self, summary_data):
        summary_file_name = f"{self._test_name}_{self._script_time}"
        with open(
            os.path.join(RESULTS_DIR_PATH, summary_file_name), "w"
        ) as summary_file:
            json.dump(summary_data, summary_file, indent=4)
            self._log.info(
                f"Results of '{self._test_name}' test case has "
                f"been saved to '{summary_file_name}'."
            )
        return summary_file_name
