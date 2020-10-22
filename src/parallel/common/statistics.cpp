#include "statistics.h"

using namespace std;

void writeStatisticsToFile(string filename) {
	create_output_dir(OUTPUT_PATH);

	string path = OUTPUT_PATH + clip_filename(filename) + "_coverage.csv";

	FILE *fp = fopen(path.c_str(), "w");

	fprintf(fp, "Step, Coverage [%%], Step time [us], Transition time [us], Find ac time [us]\n");

	for (int i = 0; i < SIMULATION_ITERATIONS; i+=STATISTICS_WRITE_FREQ) {
		fprintf(fp, "%d, %lf, %.0lf, %.0lf, %.0lf\n", i + 1,
		        stats[i].coverage,
		        stats[i].stepTime,
		        stats[i].transitionTime,
		        stats[i].findACTime);
	}
	fclose(fp);
}

void setTimeStats( Timer stepTimer, Timer transitionTimer, Timer findACTimer){
	stats->stepTime = getElapsedTime(stepTimer);
	stats->transitionTime = getElapsedTime(transitionTimer);
	stats->findACTime = getElapsedTime(findACTimer);
}
