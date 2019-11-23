import json
import logging
from json import JSONDecodeError


class TestConfigReader:
    CONFIG_PATH = "cuda_test_conf.json"

    def __init__(self):
        self._log = logging.getLogger(self.__class__.__name__)
        self._test_specs = self._parse_config()
        if not self._test_specs:
            self._log.info(f"Config file seems to be empty.")

    def _parse_config(self):
        try:
            with open(self.CONFIG_PATH, "r") as config_json:
                return self._parse_json(config_json)
        except FileNotFoundError:
            self._log.error(
                f"Could not find config file. Make sure file "
                f"'{self.CONFIG_PATH}' is in the directory of the script."
            )

    def _parse_json(self, json_file):
        try:
            return json.load(json_file)
        except JSONDecodeError:
            self._log.error(
                f"Could not parse config file: '${self.CONFIG_PATH}'"
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
            test_specs = test_details["test_specs"]
            max_length = max(*[len(l) for l in test_specs.values()])
            for name, value in test_specs.items():
                if len(value) != max_length:
                    old_length = len(value)
                    corrected_tests[test_name]["test_specs"][name] = [
                        *value,
                        *[value[-1] for _ in range(len(value), max_length)],
                    ]
                    self._log.debug(
                        f"Params list have been lengthened by "
                        f"{max_length - old_length}."
                    )
        return corrected_tests

    def _get_chosen_tests(self, test_names):
        return {
            test_name: test_specs
            for test_name, test_specs in self._test_specs.items()
            if test_name in test_names
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
                f"Could not found following test names in config "
                f"file '{self.CONFIG_PATH}': {not_found_test_names}"
            )
            exit(1)