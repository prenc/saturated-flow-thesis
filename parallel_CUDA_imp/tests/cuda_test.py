import logging
import time

from utils.ArgParser import ArgumentParser
from utils.ChartMaker import ChartMaker
from utils.TestCaseHandler import TestCaseHandler
from utils.TestConfigReader import TestConfigReader


logging.basicConfig(
    level=logging.INFO,
    filename="out.log",
    filemode="a",
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    tdp, tch, cm, args = init_program()

    test_data = tdp.get_test_data(args.cuda_tests)
    for test_name, test_params in test_data.items():
        summary_file = tch.perform_test_case(test_name, test_params)
        cm.make_chart_basing_on_summary_file(summary_file)


def init_program():
    script_start_time = time.time()
    return (
        TestConfigReader(),
        TestCaseHandler(script_start_time),
        ChartMaker(script_start_time),
        ArgumentParser().parse(),
    )


if __name__ == "__main__":
    main()
