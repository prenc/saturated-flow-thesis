import argparse


class ArgumentParser:
    def __init__(self):
        self._parser = argparse.ArgumentParser(description="Run CUDA tests")
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
            "-cd",
            "--charts_dir",
            metavar="charts_dir",
            type=str,
            nargs=1,
            help="Dir with data for charts",
        )

    def parse(self):
        return self._parser.parse_args()
