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

    def parse(self):
        return self._parser.parse_args()
