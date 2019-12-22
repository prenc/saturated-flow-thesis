import os

# dir names
COMPILED_DIR_PATH = "compiled"
PROFILING_DIR_PATH = "results/profiling"
RESULTS_DIR_PATH = "results/summaries"
CHARTS_DIR_PATH = os.getenv("CUDA_TEST_CHARTS_DIR", "charts")

TIMES_EACH_PROGRAM_IS_RUN = 5
