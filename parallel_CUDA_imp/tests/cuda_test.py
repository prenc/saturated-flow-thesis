import logging
import time

from utils.ArgParser import ArgumentParser
from utils.ChartMaker import ChartMaker
from utils.TestCaseHandler import TestCaseHandler
from utils.TestConfigReader import TestConfigReader


logging.basicConfig(level=logging.INFO)


def main():
    script_time = time.time()
    tdp = TestConfigReader()
    tch = TestCaseHandler(script_time)
    cm = ChartMaker(script_time)
    parsed_args = ArgumentParser().parse()
    test_data = tdp.get_test_data(parsed_args.cuda_tests)
    for test_name, test_params in test_data.items():
        summary_file = tch.perform_test_case(test_name, test_params)
        cm.make_charts(summary_file)


if __name__ == "__main__":
    main()
