import logging


class ResultsHandler:
    def __init__(self, test_name):
        self._log = logging.getLogger(self.__class__.__name__)

        self._test_name = test_name

    def save_results(self, run_programs):
        self._gather_results(run_programs)

    def _gather_results(self, paths):
        pass
