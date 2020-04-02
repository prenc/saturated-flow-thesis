import logging
from os import remove
from shutil import move
from tempfile import mkstemp

from ModelAnalyzer.settings import PARAMS_PATH


class DefaultParamsKeeper:
    def __init__(self):
        self._log = logging.getLogger(self.__class__.__name__)
        self.temp_file, self.temp_path = mkstemp()

    def create_params_copy(self):
        try:
            with open(PARAMS_PATH, "r") as params_file:
                self._save_default_params_in_new_file(params_file)
        except FileNotFoundError:
            self._log.error(
                f"Could not find the file with the simulation "
                f"parameters in {PARAMS_PATH}"
            )
            exit(1)

    def restore_params(self):
        remove(PARAMS_PATH)
        move(self.temp_path, PARAMS_PATH)

    def _save_default_params_in_new_file(self, filename):
        with open(self.temp_file, "w") as new_file:
            for line in filename:
                new_file.write(line)

