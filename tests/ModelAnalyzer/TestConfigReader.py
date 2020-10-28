import json
import logging
from json import JSONDecodeError

from ModelAnalyzer.settings import TEST_CONFIG_FILE


class TestConfigReader:
    def __init__(self):
        self._log = logging.getLogger(self.__class__.__name__)
        self._test_specs = self._parse_config()
        if not self._test_specs:
            self._log.warning(
                f"The config file seems to be empty: '{TEST_CONFIG_FILE}' ."
            )

    def _parse_config(self):
        try:
            with open(TEST_CONFIG_FILE, "r") as config_json:
                return self._parse_json(config_json)
        except FileNotFoundError:
            self._log.error(
                f"Could not find the config file. Make sure the file "
                f"'{TEST_CONFIG_FILE}' is in the script directory."
            )

    def _parse_json(self, json_file):
        try:
            return json.load(json_file)
        except JSONDecodeError:
            self._log.error(
                f"Could not parse the config file: '${TEST_CONFIG_FILE}'"
            )

    def get_test_data(self, test_names):
        self._log.debug(f"Provided test names: {test_names}")
        if test_names:
            chosen_tests = self._get_chosen_tests(test_names)
            self._validate_test_names(test_names, chosen_tests)
        else:
            chosen_tests = self._test_specs
        return self._make_params_list_equal_length(chosen_tests)

    def _make_params_list_equal_length(self, tests):
        corrected_tests = tests.copy()
        for test_name, test_details in tests.items():
            test_specs = test_details["params"]
            max_length = max(*[len(l) for l in test_specs.values()])
            for name, value in test_specs.items():
                if len(value) != max_length:
                    old_length = len(value)
                    corrected_tests[test_name]["params"][name] = [
                        *value,
                        *[value[-1] for _ in range(len(value), max_length)],
                    ]
                    self._log.debug(
                        f"Params list {name} has been lengthened by "
                        f"{max_length - old_length}."
                    )
        return corrected_tests

    def _get_chosen_tests(self, test_names):
        return {
            test_name: self._test_specs[test_name]
            for test_name in test_names
            if test_name in self._test_specs
        }

    def _validate_test_names(self, test_names, evaluated_test_data):
        if len(test_names) != len(evaluated_test_data):
            not_found_test_names = [
                test_name
                for test_name in test_names
                if test_name not in evaluated_test_data.keys()
            ]
            not_found_test_names = ", ".join(not_found_test_names)
            self._log.error(
                f"Could not find following test names in the config "
                f"file '{TEST_CONFIG_FILE}': {not_found_test_names}"
            )
            exit(1)
