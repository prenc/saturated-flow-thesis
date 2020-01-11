import logging
from os import remove
from shutil import move
from tempfile import mkstemp

from ModelAnalyzer.settings import PARAMS_PATH


class ParamsGenerator:
    DEFINE = "#define"

    def __init__(self, test_name):
        self._log = logging.getLogger(self.__class__.__name__)
        self._test_name = test_name

    def generate(self, test_spec):
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
        for param, value in test_spec.items():
            if f" {param.upper()} " in line:
                par = param
                val = value
                del test_spec[param]
                return self._get_define_line(par, val)
        return line

    def _get_define_line(self, var, val):
        return f"{self.DEFINE} {var.upper()} {val}\n"
