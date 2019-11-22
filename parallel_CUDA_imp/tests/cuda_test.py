import logging

from utils.ArgParser import ArgumentParser
from utils.TestCaseHandler import TestCaseHandler
from utils.TestConfigReader import TestConfigReader


logging.basicConfig(level=logging.DEBUG)


def main():
    tdp = TestConfigReader()
    tch = TestCaseHandler()
    parsed_args = ArgumentParser().parse()
    test_data = tdp.get_test_data(parsed_args.cuda_tests)
    for test_name, test_params in test_data.items():
        tch.perform_test_case(test_name, test_params)


if __name__ == "__main__":
    main()
