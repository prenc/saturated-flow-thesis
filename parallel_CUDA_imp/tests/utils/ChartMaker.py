import json
import logging
import os
from collections import defaultdict
from json import JSONDecodeError

import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import numpy

from utils.common.constants import CHARTS_DIR_PATH, RESULTS_DIR_PATH


class ChartMaker:
    def __init__(self, script_time):
        self._log = logging.getLogger(self.__class__.__name__)
        self._create_dirs()
        self.script_start_time = script_time

    @staticmethod
    def _create_dirs():
        dirs = [CHARTS_DIR_PATH]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def make_chart_basing_on_summary_file(self, summary_file):
        try:
            data = self._gather_data(summary_file)
            self._create_and_save_chart(data)
        except FileNotFoundError:
            self._log.error(f"Could not find the summary file: {summary_file}")

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
    def _create_and_save_chart(data):
        params = data["chart_params"]

        x_axis = params.get("x_axis", "ca_size")
        y_axis = "elapsed_time"

        for plot_line_name, plot_line_values in data["run_tests"].items():
            T = numpy.array(plot_line_values[x_axis])
            xnew = numpy.linspace(
                T.min(), T.max(), params.get("smooth_power", 16)
            )
            spl = make_interp_spline(
                T, numpy.array(plot_line_values[y_axis]), k=3
            )
            plt.plot(xnew, spl(xnew), "-", lw=2, label=plot_line_name)
        plt.legend()

        plt.xlabel(params.get("x_axis_label", "Cellular automata dimension"))
        plt.ylabel("Elapsed time [s]")

        plt.title(params.get("title", data["test_name"]))

        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR_PATH, f"{data['test_name']}.pdf"))
        plt.figure()

    def make_charts_in_dir(self, charts_dir):
        self._create_output_dir()
        for summary_file in os.listdir(charts_dir[0]):
            try:
                data = self._gather_data(
                    os.path.join(charts_dir[0], summary_file)
                )
                self._create_and_save_chart(data)
            except JSONDecodeError:
                self._log.warning(f"No proper json file: '{summary_file}'.")

    @staticmethod
    def _create_output_dir():
        if not os.path.exists(CHARTS_DIR_PATH):
            os.makedirs(CHARTS_DIR_PATH)
