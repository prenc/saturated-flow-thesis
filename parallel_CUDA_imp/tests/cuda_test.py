import logging

from utils.ArgParser import ArgumentParser
from utils.TestDataProvider import TestDataProvider
from utils.test_steps.ParamsGenerator import ParamsGenerator
from utils.test_steps.ProgramTester import ProgramTester

logging.basicConfig(level=logging.DEBUG)


def main():
    tdp = TestDataProvider()
    parsed_args = ArgumentParser().parse()
    test_data = tdp.get_test_data(parsed_args.cuda_tests)
    for test_name, test_params in test_data.items():
        perform_test_case(test_name, test_params)


def perform_test_case(test_name, test_params):
    pg = ParamsGenerator(test_name)

    test_specs = test_params['test_specs']
    for i in range(len(test_specs["ca_size"])):
        file_suffix = pg.generate(
            test_specs['block_size'][i],
            test_specs['ca_size'][i],
            test_specs['iterations'][i]
        )
        pt = ProgramTester(test_params['test_src'], file_suffix)
        pt.perform_test()
        # compiling program to be tested
        # testing
        # parsing test outputs and saving to one file in RESULTS_DIR_PATH
        pass


if __name__ == "__main__":
    main()
