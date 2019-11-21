import logging

from utils.ArgParser import ArgumentParser
from utils.TestDataProvider import TestDataProvider


logging.basicConfig(level=logging.DEBUG)


def main():
    tdp = TestDataProvider()
    parsed_args = ArgumentParser().parse()
    test_data = tdp.get_test_data(parsed_args.cuda_tests)


if __name__ == "__main__":
    main()
