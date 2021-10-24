#include <filesystem>
#include "statistics.h"

void save_step_stats(std::vector<StatPoint> &stats, Timer *stepTimer, int stepNumber)
{
    save_step_stats(stats, stepTimer, stepNumber, -1);
}

void save_step_stats(std::vector<StatPoint> &stats, Timer *stepTimer, int stepNumber, int devActiveCellsCount)
{
    if (stepNumber % STATISTICS_WRITE_FREQ == STATISTICS_WRITE_FREQ - 1)
    {
        stepTimer->stop();
        auto stat = new StatPoint(
                devActiveCellsCount / (double) (ROWS * COLS),
                stepTimer->elapsedNanoseconds);
        stats.push_back(*stat);
        stepTimer->start();
    }

}

void writeStatisticsToFile(std::vector<StatPoint> &stats, const std::string &filename)
{
    std::filesystem::path testName(filename);
    std::filesystem::path statsPath(OUTPUT_PATH);
    statsPath /= "coverage_";
    statsPath += testName.stem().string();
    statsPath += ".csv";
    std::filesystem::create_directories(statsPath.parent_path());

    FILE *fp = fopen(statsPath.c_str(), "w");

    fprintf(fp, "Step, Coverage [%%], Step time [ns]\n");

    auto it = stats.begin();
    for (int i{STATISTICS_WRITE_FREQ - 1}; i < SIMULATION_ITERATIONS; i += STATISTICS_WRITE_FREQ)
    {
        fprintf(fp, "%d, %lf, %.0lf\n",
                i,
                (*it).coverage,
                (*it).stepTime);
        ++it;
    }
    fclose(fp);
}

StatPoint::StatPoint(double coverage, double stepTime) :
        coverage(coverage), stepTime(stepTime)
{}
