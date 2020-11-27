import argparse


class ArgumentParser:
    def __init__(self):
        self._parser = argparse.ArgumentParser(
            description="""
            Runs model tests or creates charts based on summary files (and latex tabular code).

            Tests are run based on their definitions in TEST_CONFIG_FILE (JSON file). They are run
            in the order in which their names are passed as the script arguments.
            It is strongly recommended to set environment variable MY_DUMP to define the place where the results
            will be saved.

            Charts are only created if 'chart_params' key is specified in the summary file
            (chart_params object is passed from json config to the summary file).

            For more information check settings.py.
            """
        )
        self._add_arguments()

    def _add_arguments(self):
        self._parser.add_argument(
            "test_names",
            type=str,
            nargs="*",
            help="Runs tests with the given names from the file. If no file "
            "name is given run all from the config file.",
        )
        self._parser.add_argument(
            "-sd",
            "--summaries_dir",
            type=str,
            nargs=1,
            help="Makes charts and exports latex tables code from summary "
            "files if they have chart_params specified",
        )
        self._parser.add_argument(
            "--debug",
            action="store_true",
            help="Runs script in debug mode, not well developed",
        )
        self._parser.add_argument(
            "-c",
            "--compilation",
            action="store_true",
            help="Performs only targets compilation",
        )
        self._parser.add_argument(
            "-l",
            "--latex",
            action="store_true",
            help="Generate latex table source code files.",
        )

    def parse(self):
        return self._parser.parse_args()
