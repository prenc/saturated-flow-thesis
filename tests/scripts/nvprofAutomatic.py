#!/usr/bin/env python3

import os
from datetime import datetime


class Test:
    folderTest = ""
    executable = ""
    kernelsName = ""
    elapsedTime = ""
    testPassed = ""

    def __init__(self, folderTest, executable, kernelsName):
        self.folderTest = folderTest
        self.executable = executable
        self.kernelsName = kernelsName


class NvprofMetric:
    invocation = ""
    metricName = ""
    metricDescription = ""
    min = ""
    max = ""
    avg = ""

    def __init__(
            self, invocation, metricName, metricDescription, min, max, avg
    ):
        self.invocation = invocation
        self.metricName = metricName
        self.metricDescription = metricDescription
        self.min = min
        self.max = max
        self.avg = avg

    def __str__(self):
        strT = (
                self.invocation
                + " "
                + self.metricName
                + " "
                + self.metricDescription
                + " "
                + self.min
                + " "
                + self.max
                + " "
                + self.avg
        )
        return strT


def write_results(results, e, resultsTime, time, test_home):
    firstRow = "Metrics "
    for key in results:
        firstRow = firstRow + "," + key
    firstCol = ["Metric "]
    numberofkernels = len(results)
    for k in results:
        numberofmetrics = len(results[k])
        for m in results[k]:
            firstCol.append(results[k][m].metricName)
        break

    cols = [firstCol]
    for k in sorted(results):
        colsN = [k]
        for m in results[k]:
            colsN.append(results[k][m].avg)
        cols.append(colsN)

    out_filename = f"{test_home}/results/results_nvprof_{e}_{time}.txt"
    with open(out_filename, "w") as f:
        for k in range(numberofmetrics + 1):
            for l in range(numberofkernels + 1):
                f.write(" " + cols[l][k])
            f.write("\n")
        f.write(" time ")
        for k in sorted(resultsTime):
            f.write(" " + resultsTime[k])


def retrieve_res_from_log(e, t, metrics):
    file1 = open("./log_nvprof_" + e, "r")
    count = 0
    results = {}
    while True:
        count += 1
        line = file1.readline()
        if not line:
            break
        for k in t.kernelsName:
            if k in line:
                results[k] = {}
                line = file1.readline()
                for m in metrics:
                    if m in line:
                        line = line.split()
                        tmp = NvprofMetric(
                            line[0],
                            line[1],
                            "",
                            line[len(line) - 3],
                            line[len(line) - 2],
                            line[len(line) - 1],
                        )
                        results[k][m] = tmp
                        line = file1.readline()
    return results


def retrieve_time_from_log(e, t):
    file1 = open("./log_nvprof_time_" + e, "r")
    count = 0
    results = {}
    while True:
        count += 1
        line = file1.readline()
        if not line:
            break
        for k in t.kernelsName:
            if k in line:
                line = line.split()
                if "GPU" in line[0]:
                    results[k] = line[3]
                else:
                    results[k] = line[1]

    return results


def runNvprofAndGetOutput(tests, metrics):
    launch_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    test_home = os.getcwd()
    os.makedirs(os.path.join(test_home, "results"), exist_ok=True)

    for t in tests:
        os.chdir(t.folderTest)
        for e in t.executable:
            cmd_nvprof = f"nvprof --metrics {','.join(metrics)} ./{e} 2> ./log_nvprof_{e}"
            print("Running:", cmd_nvprof)
            os.system(cmd_nvprof)

            cmd_nvprof_time = f"nvprof --print-gpu-summary ./{e} 2> ./log_nvprof_time_{e}"
            print("Running:", cmd_nvprof_time)
            os.system(cmd_nvprof_time)

            print("Parsing results...")
            results = retrieve_res_from_log(e, t, metrics)
            res_time = retrieve_time_from_log(e, t)

            print(res_time)
            print("Saving results to file...")
            write_results(results, e, res_time, launch_time, test_home)


def main():
    tests = [
        Test(
            folderTest="../build",
            executable=["g_standard", "g_standard_stress"],
            kernelsName=["standard_step"],
        ),
        Test(
            folderTest="../build",
            executable=["g_hybrid", "g_hybrid_stress"],
            kernelsName=["hybrid_step"],
        ),
        Test(
            folderTest="../build",
            executable=["g_shared", "g_shared_stress"],
            kernelsName=["shared_step"],
        ),
        # Test(
        #     folderTest="../build",
        #     executable=["ac_g_fk", "ac_g_fk_stress"],
        #     kernelsName=["simulation_step_kernel", "findActiveCells"],
        # ),
        # Test(
        #     folderTest="../build",
        #     executable=["ac_g_sc", "ac_g_sc_stress"],
        #     kernelsName=["simulation_step_kernel"],
        # ),
        # Test(
        #     folderTest="../build",
        #     executable=["ac_g_sc_adaptive", "ac_g_sc_adaptive_stress"],
        #     kernelsName=["simulation_step_kernel", "standard_step"],
        # ),
    ]
    metrics = [
        "flop_count_dp",
        "flop_count_sp",
        "flop_count_hp",
        "gld_transactions",
        "gst_transactions",
        "l2_read_transactions",
        "l2_write_transactions",
        "dram_read_transactions",
        "dram_write_transactions",
        "sysmem_read_bytes",
        "sysmem_write_bytes",
        "shared_load_transactions",
        "shared_store_transactions",
    ]
    runNvprofAndGetOutput(tests, metrics)


if __name__ == '__main__':
    main()
