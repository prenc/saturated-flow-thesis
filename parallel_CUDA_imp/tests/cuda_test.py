import logging

from utils.ArgParser import ArgumentParser
from utils.TestDataProvider import TestDataProvider
from utils.test_steps.ParamsGenerator import ParamsGenerator
from utils.test_steps.ProgramTester import ProgramTester
from utils.test_steps.ResultsGatherer import ResultsGatherer

logging.basicConfig(level=logging.DEBUG)


def main():
    tdp = TestDataProvider()
    parsed_args = ArgumentParser().parse()
    test_data = tdp.get_test_data(parsed_args.cuda_tests)
    for test_name, test_params in test_data.items():
        perform_test_case(test_name, test_params)


def perform_test_case(test_name, test_params):
    pg = ParamsGenerator(test_name)
    pt = ProgramTester(test_params['test_src'])
    rg = ResultsGatherer(test_name)

    test_specs = test_params['test_specs']
    file_suffixes = []
    for i in range(len(test_specs["ca_size"])):
        file_suffix = pg.generate(
            test_specs['block_size'][i],
            test_specs['ca_size'][i],
            test_specs['iterations'][i]
        )
        file_suffixes.append(file_suffix)
        pt.perform_test(file_suffix)
    rg.gather_results(file_suffixes)


if __name__ == "__main__":
    main()
