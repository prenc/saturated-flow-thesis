# Author: Paweł Renc & Tomasz Pęcak
# Implementation time: 11.2019
# Last update: 9.1.2020
# for more information run script with -h option
import os

# main script settings
COMPILED_DUMP = "compiled"

"""
Compile even if previously compiled file is available
"""

ALWAYS_COMPILE = True
"""
Place to store compiled programs
"""

PROFILING_DUMP = "results/profiling"
"""
Place to store intermediate results, if the script crashes the results can be
manually restored from them (using e.g. sed :))
"""

SUMMARIES_DUMP = os.getenv("MY_DUMP", "results/summaries")
"""
Place to store final results of each test, can be set as an environment
variable for convenience
"""

CHARTS_DUMP = os.getenv("MY_DUMP", "charts")
"""
Place to store charts presenting results gather in the summary file, can be
set as an environment variable for convenience
"""

LATEX_DUMP = CHARTS_DUMP + "/latex"
"""
Place to store latex code which is latex table made based on the summary file
"""

TIMES_EACH_PROGRAM_IS_RUN = 4
"""
Defines how many times each program is run before an execution time is saved
(takes the smallest one from all results)
"""

# changing settings below is not advised
LOG_FILE = "out.log"
"""
Path to log file which is used during running tests
"""

TEST_CONFIG_FILE = "conf.json"
"""
Path to JSON file where tests are defined,
if "chart_params" is specified, a chart will be made based on the summary file
"""

PARAMS_PATH = "../src/params.h"
"""
Mandatory file, common for all CUDA C and C implementations, allows
for changing iterations number, CA dimensions, block size during running tests
"""

SRC_FILES = "../src"
"""
Path where all source files are
"""
