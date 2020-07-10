#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
struct stat st = {0};

void write_to_file(double *array, std::string fileName);
void create_output_dir(std::string path);

std::string clip_filename(std::string fullname){
	size_t lastindex = std::string(fullname).find_last_of(".");
	std::string name_without_ext = std::string(fullname).substr(0, lastindex);
	if("./" == name_without_ext.substr(0, 2)){
		return name_without_ext.substr(2, name_without_ext.length() - 2);
	}
	return name_without_ext;
}

void write_heads_to_file(double *head, std::string test_name) {
	test_name = clip_filename(test_name);
	std::string output_path= "./output/";
	create_output_dir(output_path);
	std::string fileName = output_path +
                           test_name +
                           "_" +
	                       std::to_string(CA_SIZE) +
                           "_" +
                           std::to_string(SIMULATION_ITERATIONS);
	write_to_file(head, fileName);
}

void write_river_heads_to_file(double *head, double river_head, int day) {
	std::string output_path= "./output/";
	create_output_dir(output_path);
	output_path= "./output/river/";
	create_output_dir(output_path);
	std::string fileName = output_path + std::to_string(day);
	write_to_file(head, fileName);
}


void write_coverage_to_file(double *coverage_vector, double *step_time_vector) {
	std::string output_path= "./output/";
	create_output_dir(output_path);
	std::string fileName = output_path + "coverage" +
	                       "_" +
	                       std::to_string(CA_SIZE) +
	                       "_" +
	                       std::to_string(SIMULATION_ITERATIONS) +
	                       "_" +
	                       std::to_string(NUMBER_OF_WELLS) +
	                       ".csv";

	FILE *fp;
	fp = fopen(fileName.c_str(), "w");
	fprintf(fp, "Step, Coverage [%%], Step time [us]\n");
	for (int i = COVERAGE_WRITE_FREQ - 1; i <= SIMULATION_ITERATIONS; i+=COVERAGE_WRITE_FREQ) {
		fprintf(fp, "%d, %lf, %.0lf\n", i + 1, coverage_vector[i], step_time_vector[i]);
	}
	fclose(fp);
}

void create_output_dir(std::string path){
	if (stat(path.c_str(), &st) == -1) {
		mkdir(path.c_str(), 0700);
	}
}

void write_to_file(double *head, std::string fileName) {
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
