#!/usr/bin/env python3
import logging
import sys
import time
import os

from ModelAnalyzer.ArgParser import ArgumentParser
from ModelAnalyzer.ChartMaker import ChartMaker
from ModelAnalyzer.TestCaseHandler import TestCaseHandler
from ModelAnalyzer.TestConfigReader import TestConfigReader
from ModelAnalyzer.settings import LOG_FILE, SUMMARIES_DUMP


def init_program():
    args = ArgumentParser().parse()
    if args.compilation:
        mode = TestCaseHandler.Mode.COMPILATION
    else:
        mode = TestCaseHandler.Mode.CUSTOM_PROFILING

    script_start_time = time.time()
    return (
        TestConfigReader(),
        TestCaseHandler(script_start_time, mode),
        ChartMaker(script_start_time),
        args
    )


def set_logging(args):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    if args.summaries_dir:
        fh = logging.StreamHandler(sys.stdout)
    else:
        fh = logging.FileHandler(filename=LOG_FILE, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def main():
    tcr, tch, cm, args = init_program()
    set_logging(args)

    if args.summaries_dir:
        cm.make_charts_in_dir(args.summaries_dir, args.latex)
    else:
        test_data = tcr.get_test_data(args.test_names)
        for test_name, test_params in test_data.items():
            summary_file = tch.perform_test_case(test_name, test_params)
            if not args.compilation:
                cm.make_chart_basing_on_summary_file(
                    os.path.join(SUMMARIES_DUMP, summary_file), args.latex
                )


if __name__ == "__main__":
    main()
