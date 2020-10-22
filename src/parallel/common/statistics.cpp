#include "statistics.h"

using namespace std;

void writeStatisticsToFile(string filename)
{
    create_output_dir(OUTPUT_PATH);

    string path = OUTPUT_PATH + clip_filename(filename) + "_coverage.csv";

    FILE *fp = fopen(path.c_str(), "w");

    fprintf(fp, "Step, Coverage [%%], Step time [us], Transition time [us], Find ac time [us]\n");

    for (int i{1}; i < SIMULATION_ITERATIONS; i += STATISTICS_WRITE_FREQ)
    {
        cout << i << endl;
        fprintf(fp, "%d, %lf, %.0lf, %.0lf, %.0lf\n", i,
                stats[i].coverage,
                stats[i].stepTime,
                stats[i].transitionTime,
                stats[i].findACTime);
    }
    fclose(fp);
}
