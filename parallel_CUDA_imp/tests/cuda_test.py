import logging

from utils.ArgParser import ArgumentParser
from utils.TestDataProvider import TestDataProvider
from utils.test_steps.ParamsGenerator import ParamsGenerator
from utils.test_steps.ProgramCompilerAndRunner import ProgramCompilerAndRunner
from utils.test_steps.ResultsHandler import ResultsHandler

logging.basicConfig(level=logging.DEBUG)


def main():
    tdp = TestDataProvider()
    parsed_args = ArgumentParser().parse()
    test_data = tdp.get_test_data(parsed_args.cuda_tests)
    for test_name, test_params in test_data.items():
        perform_test_case(test_name, test_params)


def perform_test_case(test_name, test_params):
    pg = ParamsGenerator(test_name)
    pcar = ProgramCompilerAndRunner(test_params["test_src"])
    rg = ResultsHandler(test_name)

    result_paths = []
    for test_spec in prepare_test_specs(test_params["test_specs"]):
        pg.generate(test_spec)
        run_tests = pcar.perform_test(test_spec)
        result_paths.extend(run_tests)
    rg.save_results(result_paths)


def prepare_test_specs(test_specs):
    return [
        {"block_size": block_size, "ca_size": ca_size, "iterations": iterations}
        for block_size, ca_size, iterations in zip(
            test_specs["block_size"],
            test_specs["ca_size"],
            test_specs["iterations"],
        )
    ]


if __name__ == "__main__":
    main()
