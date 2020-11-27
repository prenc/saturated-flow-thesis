import json
import logging
import os
from collections import defaultdict
from itertools import cycle
from json import JSONDecodeError
from datetime import datetime

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

from ModelAnalyzer.settings import CHARTS_DUMP, LATEX_DUMP

plt.style.use("grayscale")


class ChartMaker:
    def __init__(self, script_time):
        self._log = logging.getLogger(self.__class__.__name__)
        self._create_dirs([CHARTS_DUMP])
        self.script_start_time = script_time
        current_date = datetime.strftime(datetime.now(), "%y%m%d_%H%M")
        self._chart_file = os.path.join(CHARTS_DUMP, f"{current_date}.pdf")
        self._pdf = matplotlib.backends.backend_pdf.PdfPages(self._chart_file)

    def __del__(self):
        self._pdf.close()

    @staticmethod
    def _create_dirs(dirs):
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def make_chart_basing_on_summary_file(self, summary_path, latex=False):
        try:
            data = self._gather_data(summary_path)
            if "chart_params" in data.keys():
                self._create_and_save_chart(data, latex)
        except FileNotFoundError:
            self._log.error(f"Could not find the summary file: {summary_path}")
            exit(1)
        except (JSONDecodeError, IsADirectoryError, UnicodeDecodeError):
            self._log.warning(f"No proper json file: '{summary_path}'. ")

    def _gather_data(self, summary_file):
        data = self._load_charts_data(summary_file)
        return self._prepare_data(data)

    @staticmethod
    def _load_charts_data(file_path):
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
        params = data["chart_params"]

        x_axis = params.get("x_axis", "ca_size")
        y_axis = params.get("y_axis", "elapsed_time")

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        line_styles = cycle([":", "--", "-.", "-"])
        marker_styles = cycle(("o", "x", "*", "+", "D", "X", ".", "s"))
        for plot_line_name, plot_values in data["run_tests"].items():
            x, y = plot_values[x_axis], plot_values[y_axis]

            if smooth_power := params.get("smooth_power", None):
                x, y = self._smoothen_plot_line(x, y, smooth_power)

            ax.plot(
                x,
                y,
                linestyle=next(line_styles),
                marker=next(marker_styles),
                lw=1,
                label=plot_line_name.replace("_", " ").title(),
            )

        ax.set_xlabel(params.get("x_axis_label", "CA dimensions"))
        ax.set_ylabel(params.get("y_axis_label", "Elapsed time [s]"))

        ax.set_title(params.get("title", data["test_name"]))

        ax.grid(False)
        ax.legend()

        self._pdf.savefig(fig)
        self._log.info(
            f"Chart '{params.get('title', data['test_name'])}' has been added"
            f" in '{self._chart_file}'"
        )
        plt.close(fig)

        if latex:
            self._create_dirs([LATEX_DUMP])
            self._create_latex_tabular_file(data, x_axis, y_axis)

    @staticmethod
    def _smoothen_plot_line(x, y, smooth_power):
        new_x = np.linspace(min(x), max(x), smooth_power)
        spl = make_interp_spline(np.array(new_x), np.array(y))
        return spl(new_x), y

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

    def make_charts_in_dir(self, charts_dir, latex=False):
        self._log.info(f"Saving all charts in '{CHARTS_DUMP}'")
        if latex:
            self._log.info(f"Saving all latex tables in '{LATEX_DUMP}'")
        for summary_file in sorted(os.listdir(charts_dir[0])):
            summary_file_path = os.path.join(charts_dir[0], summary_file)
            self.make_chart_basing_on_summary_file(
                summary_file_path, latex=latex
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
