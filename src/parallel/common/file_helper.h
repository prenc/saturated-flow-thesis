#ifndef SATURATED_FLOW_THESIS_FILE_HELPER_H
#define SATURATED_FLOW_THESIS_FILE_HELPER_H

#include <iostream>
#include "../../params.h"

void saveHeadsInFile(double *&head, char *test_name);

void saveRiverHeadsInFile(double *&head, double &river_head, int &day);

void writeHeads(double *&heads, const std::string& file_path);

#endif //SATURATED_FLOW_THESIS_FILE_HELPER_H
