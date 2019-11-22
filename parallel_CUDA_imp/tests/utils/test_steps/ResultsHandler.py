import json
import logging
import os

from utils.common.constants import RESULTS_DIR_PATH


class ResultsHandler:
    def __init__(self, test_name, time):
        self._log = logging.getLogger(self.__class__.__name__)
        self.time = time
        self._test_name = test_name

    def save_results(self, run_programs):
        data = self._gather_results(run_programs)
        self._save_summary_to_file(data)

    @staticmethod
    def _gather_results(run_programs_paths):
        summary_results = {}
        for path in run_programs_paths:
            with open(path, "r") as result_file:
                result_json = json.load(result_file)
                src_name = result_json["src_name"].split(".")[0]
                if src_name in summary_results.keys():
                    summary_results[src_name] = (
                        [*summary_results[src_name], result_json]
                    )
                else:
                    summary_results[src_name] = [result_json]
        return summary_results

    def _save_summary_to_file(self, summary_data):
        with open(
            os.path.join(RESULTS_DIR_PATH, f"{self._test_name}_{self.time}"),
            "w",
        ) as summary_file:
            json.dump(summary_data, summary_file, indent=4)
