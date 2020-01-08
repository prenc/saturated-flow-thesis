# Author: Paweł Renc
# Implementation time: 11.2019
# Last update: 8.1.2020
import os

# dir names
COMPILED_DUMP = "compiled"
"""
Place to store compiled programs
"""

PROFILING_DUMP = "results/profiling"
"""
Place to store intermediate results, if the script crashes the results can be 
manually restored from them (using e.g. sed :))
"""

SUMMARIES_DUMP = os.getenv("CUDA_TEST_DUMP", "results/summaries")
"""
Place to store final results of each test, can be set as an environment 
variable for convenience
"""

CHARTS_DUMP = os.getenv("CUDA_TEST_DUMP", "charts")
"""
Place to store charts presenting results gather in the summary file, can be 
set as an environment variable for convenience
"""

LATEX_DUMP = CHARTS_DUMP + "/latex"
"""
Place to store latex code which is latex table made based on the summary file
"""

PARAMS_PATH = "../src/params.h"
"""
Mandatory file, common for all CUDA C and C implementations, allows to change 
iterations number, CA dimensions, block size when testing
"""

TIMES_EACH_PROGRAM_IS_RUN = 3
"""
Defines how many times each program is run before an execution time is saved 
(takes the smallest one from all results)
"""

TEST_CONFIG_FILE = "cuda_test_conf.json"
"""
Path to JSON file where tests are defined,
if "chart_params" is specified, a chart will be made based on the summary 
file 
"""

LOG_FILE = "out.log"
"""
Path to log file which is used during running tests
"""
