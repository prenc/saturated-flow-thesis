#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
struct stat st = {0};

using namespace std;

void write_to_file(double *array, string fileName);
void create_output_dir(string path);

string OUTPUT_PATH = "output/";

string clip_filename(string fullname){
    string name_without_ext = fullname;

    size_t last_slash_idx = name_without_ext.find_last_of("/");
    if (last_slash_idx != string::npos) {
        name_without_ext = name_without_ext.substr(last_slash_idx + 1);
    }

    size_t last_dot_idx = name_without_ext.find_last_of(".");
    if (last_dot_idx != string::npos) {
        name_without_ext = name_without_ext.substr(0, last_dot_idx);
    }

    return name_without_ext;
}

void write_heads_to_file(double *head, string test_name) {
    create_output_dir(OUTPUT_PATH);

    string fileName = OUTPUT_PATH + clip_filename(test_name) + "_heads" + 
        "_" + to_string(CA_SIZE) +
        "_" + to_string(SIMULATION_ITERATIONS) + 
        ".csv";

    write_to_file(head, fileName);
}

void write_river_heads_to_file(double *head, double river_head, int day) {
    string output_path = "./output/river/";
    create_output_dir(output_path);
    string fileName = output_path + to_string(day);
    write_to_file(head, fileName);
}


void write_coverage_to_file(double *coverage_vector, double *step_time_vector, string filename) {

    create_output_dir(OUTPUT_PATH);

    string fileName = OUTPUT_PATH + clip_filename(filename) + "_coverage" +
        "_" + to_string(CA_SIZE) +
        "_" + to_string(SIMULATION_ITERATIONS) +
        "_" + to_string(NUMBER_OF_WELLS) +
        ".csv";

    FILE *fp = fopen(fileName.c_str(), "w");
    fprintf(fp, "Step, Coverage [%%], Step time [us]\n");

    for (int i = COVERAGE_WRITE_FREQ - 1; i <= SIMULATION_ITERATIONS; i+=COVERAGE_WRITE_FREQ) {
        fprintf(fp, "%d, %lf, %.0lf\n", i + 1, coverage_vector[i], step_time_vector[i]);
    }
    fclose(fp);
}

void create_output_dir(string path){
    if (stat(path.c_str(), &st) == -1) {
        mkdir(path.c_str(), 0770);
    }
}

void write_to_file(double *head, string fileName) {
    FILE *fp;
    fp = fopen(fileName.c_str(), "w");

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if(TRANSPOSE_OUTPUT == 1){
                fprintf(fp, "%lf, ", head[j * ROWS + i]);
            }else{
                fprintf(fp, "%lf, ", head[i * ROWS + j]);
            }
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}
