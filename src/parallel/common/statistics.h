#ifndef SATURATED_FLOW_THESIS_STATISTICS_H
#define SATURATED_FLOW_THESIS_STATISTICS_H

#include <string>

#include "timer.h"
#include "../../params.h"
#include "file_helper.h"

using namespace std;

struct Statistics
{
    double coverage;
    double stepTime;
    double transitionTime;
    double findACTime;
};

static Statistics stats[SIMULATION_ITERATIONS];

void writeStatisticsToFile(string filename);

#endif //SATURATED_FLOW_THESIS_STATISTICS_H
