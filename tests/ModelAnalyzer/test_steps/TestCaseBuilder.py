import subprocess
from shutil import rmtree

from ModelAnalyzer.settings import (
    CMAKE_LISTS_PATH,
    CMAKE_BUILD_DIR,
)


class TestCaseBuilder:
    @staticmethod
    def build_test():
        subprocess.run(
            [
                "cmake",
                f"-B{CMAKE_BUILD_DIR}",
                f"-H{CMAKE_LISTS_PATH}",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_MODULE_PATH=/nfshome/aderango/git/opencal/cmake",
                "-DOpenCL_LIBRARY=/opt/cuda/targets/x86_64-linux/lib/libOpenCL.so",
                "-DOpenCL_INCLUDE_DIR=/opt/cuda/targets/x86_64-linux/include",
            ]
        )

    @staticmethod
    def clean_build():
        rmtree(CMAKE_BUILD_DIR, ignore_errors=True)
