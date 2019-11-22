import json
import logging
import os

from utils.common.constants import RESULTS_DIR_PATH


class ResultsHandler:
    def __init__(self, test_name, script_time):
        self._log = logging.getLogger(self.__class__.__name__)
        self._script_time = script_time
        self._test_name = test_name

    def save_results(self, run_programs, test_case_time):
        data = self._gather_results(run_programs, test_case_time)
        self._save_summary_to_file(data)
        self._log.info(
            f"Test case took {test_case_time // 60}m"
            f"{test_case_time % 60:.0f}s"
        )

    def _gather_results(self, run_programs_paths, test_case_time):
        summary_results = {
            "test_case": self._test_name,
            "test_case_time": test_case_time,
            "run_tests": {},
        }
        for path in run_programs_paths:
            with open(path, "r") as result_file:
                result_json = json.load(result_file)
                src_name = result_json["src_name"].split(".")[0]
                if src_name in summary_results["run_tests"].keys():
                    summary_results["run_tests"][src_name] = [
                        *summary_results["run_tests"][src_name],
                        result_json,
                    ]
                else:
                    summary_results["run_tests"][src_name] = [result_json]
        return summary_results

    def _save_summary_to_file(self, summary_data):
        summary_file_name = f"{self._test_name}_{self._script_time}"
        with open(
            os.path.join(RESULTS_DIR_PATH, summary_file_name), "w"
        ) as summary_file:
            json.dump(summary_data, summary_file, indent=4)
            self._log.info(
                f"Results of '{self._test_name}' test case has "
                f"been saved to '{summary_file_name}'."
            )
