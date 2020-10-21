#ifndef SATURATED_FLOW_THESIS_FILE_HELPER_H
#define SATURATED_FLOW_THESIS_FILE_HELPER_H

#include <iostream>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../../params.h"

static struct stat st = {0};

using namespace std;

string clip_filename(string fullname);
void write_heads_to_file(double *head, string test_name);
void write_river_heads_to_file(double *head, double river_head, int day);
void create_output_dir(string path);
void write_to_file(double *head, string filename);

#endif //SATURATED_FLOW_THESIS_FILE_HELPER_H
