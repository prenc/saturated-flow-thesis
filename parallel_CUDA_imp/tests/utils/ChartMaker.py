import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt

from utils.common.constants import CHARTS_DIR_PATH, RESULTS_DIR_PATH


class ChartMaker:
    def __init__(self, script_time):
        self._create_dirs()
        self.script_start_time = script_time

    @staticmethod
    def _create_dirs():
        dirs = [CHARTS_DIR_PATH]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def make_charts(self, summary_file):
        data = self._gather_data(summary_file)
        self._create_and_save_charts(data)

    def _gather_data(self, summary_file):
        data = self._load_charts_data(summary_file)
        return self._prepare_data(data)

    @staticmethod
    def _load_charts_data(data_file):
        file_path = os.path.join(RESULTS_DIR_PATH, data_file)
        with open(file_path, "r") as json_file:
            return json.load(json_file)

    @staticmethod
    def _prepare_data(data, sort="ca_size"):
        new_data = data.copy()
        for name, test_results in data["run_tests"].items():
            sorted_test_results = sorted(test_results, key=lambda x: x[sort])
            new_data["run_tests"][name] = defaultdict(list)
            for test_result in sorted_test_results:
                for key, value in test_result.items():
                    new_data["run_tests"][name][key].append(value)
        return new_data

    @staticmethod
    def _create_and_save_charts(data):
        x_axis = "ca_size"
        y_axis = "elapsed_time"
        for plot_line_name, plot_line_values in data["run_tests"].items():
            plt.plot(
                plot_line_values[x_axis],
                plot_line_values[y_axis],
                "-",
                lw=2,
                label=plot_line_name,
            )
        plt.legend()

        plt.xlabel(y_axis)
        plt.ylabel(x_axis)
        plt.title(data["test_name"])
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR_PATH, f"{data['test_name']}.pdf"))
        plt.figure()