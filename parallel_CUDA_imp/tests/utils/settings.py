import os

# dir names
COMPILED_DIR_PATH = "compiled"
PROFILING_DIR_PATH = "results/profiling"
RESULTS_DIR_PATH = os.getenv("CUDA_TEST_DUMP", "results/summaries")
CHARTS_DIR_PATH = os.getenv("CUDA_TEST_DUMP", "charts")
LATEX_FILE_DUMP = CHARTS_DIR_PATH + "/latex"
PARAMS_PATH = "../src/params.h"

TIMES_EACH_PROGRAM_IS_RUN = 3
