import argparse


class ArgumentParser:
    def __init__(self):
        self._parser = argparse.ArgumentParser(
            description="Run model tests, it is strongly recommended to set "
            "environment variable CUDA_TEST_DUMP to idicate place where "
            "result will be placed"
        )
        self._add_arguments()

    def _add_arguments(self):
        self._parser.add_argument(
            "cuda_tests",
            metavar="test_name",
            type=str,
            nargs="*",
            help="cuda test names to run",
        )
        self._parser.add_argument(
            "-sd",
            "--summaries_dir",
            metavar="summaries dir",
            type=str,
            nargs=1,
            help="Dir with data to make charts",
        )

    def parse(self):
        return self._parser.parse_args()
