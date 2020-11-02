# Author: Paweł Renc & Tomasz Pęcak
# Date of the first implemenatation: 11.2019
# for more information run script with -h option
import os

# main script settings
COMPILED_DUMP = "compiled"

"""
Compile even if previously compiled file is available
"""

ALWAYS_COMPILE = True
"""
Alawys recompile models, even if they are already present
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

TIMES_EACH_PROGRAM_IS_RUN = 5
"""
Defines how many times each program is run before an execution time is saved
(takes the smallest one from all results)
"""

LOG_FILE = "out.log"
"""
Path to log file which is used during running tests
"""

PARAMS_PATH = "../src/params.h"
"""
Mandatory file, common for all CUDA C and C implementations, allows
for changing iterations number, CA dimensions, block size during running tests
"""

CMAKE_LISTS_PATH = "../src"
"""
Path where main project CMakeLists.txt is stored.
"""

CMAKE_BUILD_DIR = "../src/build"
"""
Path to cmake build directory. It is removed after test.  
"""

TEST_CONFIG_FILE = "conf.json"
"""
Path to JSON file where tests are defined.
Each test is a separate object and has two keys params and targets.
Targets value is a list of targets names from {CMAKE_PATH}/CmakeLists.txt.    
Params value is an object which keys names correspond to test params from {PARAMS_PATH} file.    
Each param value is a list. Targets are executed with all params lists permutations.   
If "chart_params" is specified, a chart will be made based on the summary file
"""