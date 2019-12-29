import json
import logging
import os
from collections import defaultdict
from json import JSONDecodeError

import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np

from utils.settings import CHARTS_DUMP, RESULTS_DIR_PATH, LATEX_DUMP


class ChartMaker:
    def __init__(self, script_time):
        self._log = logging.getLogger(self.__class__.__name__)
        self._create_dirs()
        self.script_start_time = script_time

    @staticmethod
    def _create_dirs():
        dirs = [CHARTS_DUMP, LATEX_DUMP]
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

    def _create_and_save_chart(self, data):
        params = data["chart_params"]

        x_axis = params.get("x_axis", "ca_size")
        y_axis = "elapsed_time"

        for plot_line_name, plot_line_values in data["run_tests"].items():
            self._create_plot_line(
                {
                    "plot_line_name": plot_line_name,
                    "x_values": plot_line_values[x_axis],
                    "y_values": plot_line_values[y_axis],
                    "smooth_power": params.get("smooth_power", 16),
                }
            )
        plt.legend()

        plt.xlabel(params.get("x_axis_label", "Cellular automata dimension"))
        plt.ylabel("Elapsed time [s]")

        plt.title(params.get("title", data["test_name"]))

        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DUMP, f"{data['test_name']}.pdf"))
        plt.figure()

        self._make_latex_tabular(data, x_axis, y_axis)

    @staticmethod
    def _create_plot_line(plot_params):
        if plot_params["smooth_power"]:
            T = np.array(plot_params["x_values"])
            x_smoothed = np.linspace(
                T.min(), T.max(), plot_params["smooth_power"]
            )
            spl = make_interp_spline(T, np.array(plot_params["y_values"]), k=3)
            plt.plot(
                x_smoothed,
                spl(x_smoothed),
                "-",
                lw=2,
                label=plot_params["plot_line_name"],
            )
        else:
            plt.plot(
                plot_params["x_values"],
                plot_params["y_values"],
                "-",
                lw=2,
                label=plot_params["plot_line_name"],
            )

    def _make_latex_tabular(self, data, x_axis, y_axis):
        tabular_values = [
            ["names", *next(iter(data["run_tests"].values()))[x_axis]]
        ]
        for plot_line_name, plot_line_values in data["run_tests"].items():
            tabular_values.append(
                [
                    plot_line_name,
                    *[round(value, 2) for value in plot_line_values[y_axis]],
                ]
            )
        tabular_output = ""
        for row_values in zip(*tabular_values):
            tabular_output += " & ".join([str(value) for value in row_values])
            tabular_output += " \\\\\n"
        with open(
            os.path.join(LATEX_DUMP, f"{data['test_name']}_latex"), "w"
        ) as latex_table_file:
            latex_table_file.write(tabular_output)

    def make_charts_in_dir(self, charts_dir):
        for summary_file in os.listdir(charts_dir[0]):
            try:
                data = self._gather_data(
                    os.path.join(charts_dir[0], summary_file)
                )
                self._create_and_save_chart(data)
            except JSONDecodeError:
                self._log.warning(f"No proper json file: '{summary_file}'.")
