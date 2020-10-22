#include "statistics.h"

void writeStatisticsToFile(vector<StatPoint> &stats, string filename)
{
    create_output_dir(OUTPUT_PATH);

    string path = OUTPUT_PATH + clip_filename(filename) + "_coverage.csv";

    FILE *fp = fopen(path.c_str(), "w");

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
