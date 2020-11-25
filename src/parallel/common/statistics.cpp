#include <filesystem>
#include "statistics.h"

void writeStatisticsToFile(std::vector<StatPoint> &stats, const std::string &filename)
{
    std::filesystem::path testName(filename);
    std::filesystem::path statsPath(OUTPUT_PATH);
    statsPath /= "coverage_";
    statsPath += testName.stem().string();
    statsPath += ".csv";
    std::filesystem::create_directories(statsPath.parent_path());

    FILE *fp = fopen(statsPath.c_str(), "w");

    fprintf(fp, "Step, Coverage [%%], Step time [ns], Transition time [ns], Find ac time [ns]\n");

    auto it = stats.begin();
    for (int i{STATISTICS_WRITE_FREQ - 1}; i < SIMULATION_ITERATIONS; i += STATISTICS_WRITE_FREQ)
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

StatPoint::StatPoint(double coverage, double stepTime, double transitionTime, double findACTime) :
        coverage(coverage), stepTime(stepTime), transitionTime(transitionTime),
        findACTime(findACTime)
{}
