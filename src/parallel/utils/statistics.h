#ifndef SATURATED_FLOW_THESIS_STATISTICS_H
#define SATURATED_FLOW_THESIS_STATISTICS_H

#include <string>
#include <vector>

#include "timer.h"
#include "../../params.h"
#include "file_helper.h"

struct StatPoint
{
    double coverage;
    double stepTime;

    StatPoint() = default;
    StatPoint(double coverage, double stepTime);
};

void save_step_stats(std::vector<StatPoint> &stats, Timer *stepTimer, int stepNumber);
void save_step_stats(std::vector<StatPoint> &stats, Timer *stepTimer, int stepNumber, int devActiveCellsCount);

void writeStatisticsToFile(std::vector<StatPoint> &stats, const std::string& filename);

#endif //SATURATED_FLOW_THESIS_STATISTICS_H
