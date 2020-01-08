#!/usr/bin/env python3
import logging
import sys
import time

from utils.ArgParser import ArgumentParser
from utils.ChartMaker import ChartMaker
from utils.TestCaseHandler import TestCaseHandler
from utils.TestConfigReader import TestConfigReader
from utils.settings import LOG_FILE


def init_program():
    script_start_time = time.time()
    return (
        TestConfigReader(),
        TestCaseHandler(script_start_time),
        ChartMaker(script_start_time),
        ArgumentParser().parse(),
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
    tdp, tch, cm, args = init_program()
    set_logging(args)

    if args.summaries_dir:
        cm.make_charts_in_dir(args.summaries_dir)
    else:
        test_data = tdp.get_test_data(args.test_names)
        for test_name, test_params in test_data.items():
            summary_file = tch.perform_test_case(test_name, test_params)
            cm.make_chart_basing_on_summary_file(summary_file)


if __name__ == "__main__":
    main()
