import logging

from utils.ArgParser import ArgumentParser
from utils.TestCaseHandler import TestCaseHandler
from utils.TestDataProvider import TestDataProvider


logging.basicConfig(level=logging.INFO)


def main():
    tdp = TestDataProvider()
    tch = TestCaseHandler()
    parsed_args = ArgumentParser().parse()
    test_data = tdp.get_test_data(parsed_args.cuda_tests)
    for test_name, test_params in test_data.items():
        tch.perform_test_case(test_name, test_params)


if __name__ == "__main__":
    main()
