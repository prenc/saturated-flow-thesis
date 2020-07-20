import logging
import itertools
from os import remove
from shutil import move
from tempfile import mkstemp

from ModelAnalyzer.settings import PARAMS_PATH


class ParamsGenerator:
    DEFINE = "#define"
    WELLS_NUMBER = "NUMBER_OF_WELLS"

    def __init__(self, test_name):
        self._log = logging.getLogger(self.__class__.__name__)
        self._test_name = test_name

    def generate(self, test_spec):
        test_spec = self._handle_wells(test_spec)
        try:
            with open(PARAMS_PATH, "r") as params_file:
                self._modify_file(params_file, test_spec)
        except FileNotFoundError:
            self._log.error(
                f"Could not find the file with the simulation "
                f"parameters in {PARAMS_PATH}"
            )
            exit(1)

    def _modify_file(self, filename, test_spec):
        test_spec_c = test_spec.copy()
        temp_file, temp_path = mkstemp()
        with open(temp_file, "w") as new_file:
            self._change_lines_with_defines(filename, test_spec_c, new_file)
            self._add_rest_defines_line(test_spec_c, new_file)
        remove(PARAMS_PATH)
        move(temp_path, PARAMS_PATH)

    def _change_lines_with_defines(self, filename, test_spec, new_file):
        for line in filename:
            new_file.write(self._modify_line(line, test_spec))

    def _add_rest_defines_line(self, test_spec, new_file):
        for param, value in test_spec.items():
            new_file.write(self._get_define_line(param, value))

    def _modify_line(self, line, test_spec):
        params_to_be_removed = list()
        for param, value in test_spec.items():
            if f" {param.upper()} " in line:
                del test_spec[param]
                return self._get_define_line(param, value)
        return line

    def _get_define_line(self, var, val):
        if isinstance(val, tuple) or isinstance(val, list):
            value = str(val)[1:-1]
        else:
            value = val
        return f"{self.DEFINE} {var.upper()} {value}\n"

    def _handle_wells(self, test_spec) -> list:
        updated_test_spec = {k.upper(): v for k, v in test_spec.items()}

        if self.WELLS_NUMBER in updated_test_spec:
            number_of_wells = updated_test_spec[self.WELLS_NUMBER]
            ca_size = updated_test_spec["CA_SIZE"]

            dist_bwn_wells = round(ca_size / number_of_wells)
            wells_locs = list(
                range(
                    round(dist_bwn_wells / 2),
                    dist_bwn_wells * number_of_wells,
                    dist_bwn_wells,
                )
            )

            wells_xs, wells_ys = zip(
                *itertools.product(wells_locs, wells_locs)
            )
            updated_test_spec["WELLS_Y"] = wells_ys
            updated_test_spec["WELLS_X"] = wells_xs

        return updated_test_spec
