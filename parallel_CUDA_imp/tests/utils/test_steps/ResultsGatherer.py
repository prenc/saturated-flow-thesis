import logging


class ResultsGatherer:
    def __init__(self, test_name):
        self._log = logging.getLogger(self.__class__.__name__)

        self._test_name = test_name

    def gather_results(self, suffixes:list):
        pass
