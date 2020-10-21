#ifndef SATURATED_FLOW_THESIS_STATISTICS_H
#define SATURATED_FLOW_THESIS_STATISTICS_H

#include <string>

#include "timer.h"
#include "../../params.h"
#include "file_helper.h"

using namespace std;

struct Statistics{
		double coverage;
		double stepTime;
		double transitionTime;
		double findACTime;
};

static Statistics stats[SIMULATION_ITERATIONS];

void write_statistics_to_file( string filename);
void setTimeStats( Timer stepTimer, Timer transitionTimer, Timer findACTimer);

#endif //SATURATED_FLOW_THESIS_STATISTICS_H
