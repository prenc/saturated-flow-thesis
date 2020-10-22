#include <filesystem>
#include "statistics.h"

void writeStatisticsToFile(std::vector<StatPoint> &stats, const std::string& filename)
{
    std::filesystem::path testName(filename);
    std::filesystem::path statsPath(OUTPUT_PATH);
    statsPath /= "coverage_";
    statsPath += testName.stem().string();
    statsPath += ".csv";
    std::filesystem::create_directories(statsPath.parent_path());

    FILE *fp = fopen(statsPath.c_str(), "w");

    fprintf(fp, "Step, Coverage [%%], Step time [us], Transition time [us], Find ac time [us]\n");

    auto it = stats.begin();
    for (int i{1}; i < SIMULATION_ITERATIONS; i += STATISTICS_WRITE_FREQ)
    {
        fprintf(fp, "%d, %lf, %.0lf, %.0lf, %.0lf\n",
                i,
                (*it).coverage,
                (*it).stepTime,
                (*it).transitionTime,
                (*it).findACTime);
        ++it;
    }
    fclose(fp);
}
