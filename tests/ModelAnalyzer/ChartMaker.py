import json
import logging
import os
from collections import defaultdict
from json import JSONDecodeError

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

from ModelAnalyzer.settings import CHARTS_DUMP, SUMMARIES_DUMP, LATEX_DUMP


class ChartMaker:
    def __init__(self, script_time):
        self._log = logging.getLogger(self.__class__.__name__)
        self._create_dirs([CHARTS_DUMP])
        self.script_start_time = script_time

    @staticmethod
    def _create_dirs(dirs):
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def make_chart_basing_on_summary_file(self, summary_file, latex=False):
        try:
            data = self._gather_data(summary_file)
            if "chart_params" in data.keys():
                self._create_and_save_chart(data, latex)
        except FileNotFoundError:
            self._log.error(f"Could not find the summary file: {summary_file}")
            exit(1)
        except (JSONDecodeError, IsADirectoryError, UnicodeDecodeError):
            self._log.warning(f"No proper json file: '{summary_file}'.")

    def _gather_data(self, summary_file):
        data = self._load_charts_data(summary_file)
        return self._prepare_data(data)

    @staticmethod
    def _load_charts_data(data_file):
        file_path = os.path.join(SUMMARIES_DUMP, data_file)
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

    def _create_and_save_chart(self, data, latex=False):
        plt.style.use("grayscale")
        params = data["chart_params"]

        x_axis = params.get("x_axis", "ca_size")
        y_axis = params.get("y_axis", "elapsed_time")

        line_styles = iter([":", "--", "-.", "-"] * 3)
        for plot_line_name, plot_line_values in data["run_tests"].items():
            self._create_plot_line(
                {
                    "plot_line_name": plot_line_name[0].upper()
                    + plot_line_name[1:].replace("_", " "),
                    "x_values": plot_line_values[x_axis],
                    "y_values": plot_line_values[y_axis],
                    "smooth_power": params.get("smooth_power", None),
                    "line_style": next(line_styles),
                    "line_width": 2,
                }
            )
        plt.legend()

        plt.xlabel(params.get("x_axis_label", "CA dimensions"))
        plt.ylabel(params.get("y_axis_label", "Elapsed time [s]"))

        plt.title(params.get("title", data["test_name"]))

        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DUMP, f"{data['test_name']}.pdf"))
        self._log.info(f"Chart has been created: {data['test_name']}.pdf")
        plt.figure()

        if latex:
            self._create_dirs([LATEX_DUMP])
            self._create_latex_tabular_file(data, x_axis, y_axis)

    @staticmethod
    def _create_plot_line(plot_params):
        if plot_params["smooth_power"]:
            new_xs = np.array(plot_params["x_values"])
            x_values = np.linspace(
                new_xs.min(), new_xs.max(), plot_params["smooth_power"]
            )
            spl = make_interp_spline(
                new_xs, np.array(plot_params["y_values"]), k=3
            )
            y_values = spl(x_values)
        else:
            x_values = plot_params["x_values"]
            y_values = plot_params["y_values"]

        plt.plot(
            x_values,
            y_values,
            linestyle=plot_params["line_style"],
            lw=plot_params["line_width"],
            label=plot_params["plot_line_name"],
        )

    def _create_latex_tabular_file(self, data, x_axis, y_axis):
        first_col_values = self._get_longest_first_column(data, x_axis)
        b = "\\"
        tabular_values = [["CA dimensions", *first_col_values]]
        for plot_line_name, plot_line_values in data["run_tests"].items():
            tabular_values.append(
                [
                    f"{b}makecell{{{plot_line_name.replace('_', b*2)}}}",
                    *[
                        f"\SI{{{round(value, 2):.2f}}}{{{b}second}}"
                        for value in plot_line_values[y_axis]
                    ],
                ]
            )
        tabular_values = self._make_lists_eq("x", *tabular_values)
        tabular_output = ""
        for row_values in zip(*tabular_values):
            tabular_output += " & ".join([str(value) for value in row_values])
            tabular_output += " \\\\\\hline\n"
        with open(
            os.path.join(LATEX_DUMP, f"{data['test_name']}_latex"), "w"
        ) as latex_table_file:
            latex_table_file.write(tabular_output)
        self._log.info(
            f"Latex tabular code has been saved: {data['test_name']}_latex"
        )

    @staticmethod
    def _get_longest_first_column(data, x_axis):
        first_columns = [
            results[x_axis] for results in data["run_tests"].values()
        ]
        max_l = max([len(col) for col in first_columns])
        for col in first_columns:
            if len(col) == max_l:
                return col

    def make_charts_in_dir(self, charts_dir):
        for summary_file in os.listdir(charts_dir[0]):
            summary_file_path = os.path.join(charts_dir[0], summary_file)
            self.make_chart_basing_on_summary_file(
                summary_file_path, latex=True
            )

    @staticmethod
    def _make_lists_eq(filler, *lists) -> list:
        max_len = max([len(lst) for lst in lists])
        out_lsts = []
        for lst in lists:
            if len(lst) < max_len:
                out_lsts.append(lst + [filler] * (max_len - len(lst)))
            else:
                out_lsts.append(lst)
        return out_lsts
