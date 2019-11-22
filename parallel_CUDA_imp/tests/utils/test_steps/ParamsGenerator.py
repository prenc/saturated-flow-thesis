import logging
from os import remove
from shutil import move
from tempfile import mkstemp


class ParamsGenerator:
    PARAMS_PATH = "../src/params.h"
    MATCH_BLOCK_SIZE = "#define BLOCK_SIZE "
    MATCH_CA_SIZE = "#define CA_SIZE "
    MATCH_ITERATIONS = "#define SIMULATION_ITERATIONS "

    def __init__(self, test_name):
        self._log = logging.getLogger(self.__class__.__name__)
        self._test_name = test_name

    def generate(self, test_spec):
        try:
            with open(self.PARAMS_PATH, "r") as params_file:
                self._modify_file(params_file, test_spec)
        except FileNotFoundError:
            self._log.error(
                f"Could not find the file with simulation "
                f"parameters: {self.PARAMS_PATH}"
            )
            exit(1)

    def _modify_file(self, filename, test_spec):
        temp_file, temp_path = mkstemp()
        with open(temp_file, "w") as new_file:
            for line in filename:
                new_line = self._modify_line(
                    line, **test_spec
                )
                new_file.write(new_line)
        remove(self.PARAMS_PATH)
        move(temp_path, self.PARAMS_PATH)

    def _modify_line(self, line, block_size, ca_size, iterations):
        if self.MATCH_BLOCK_SIZE in line:
            return f"{self.MATCH_BLOCK_SIZE}{block_size}\n"
        elif self.MATCH_CA_SIZE in line:
            return f"{self.MATCH_CA_SIZE}{ca_size}\n"
        elif self.MATCH_ITERATIONS in line:
            return f"{self.MATCH_ITERATIONS}{iterations}\n"
        else:
            return line
