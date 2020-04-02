#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
struct stat st = {0};

void write_heads_to_file(double *head, std::string test_name) {
	if (stat("output", &st) == -1) {
		mkdir("./output", 0700);
	}
	std::string fileName = "./output/"+test_name+"_"+std::to_string(BLOCK_SIZE)+"_"
	                       +std::to_string(CA_SIZE)+"_"+std::to_string(SIMULATION_ITERATIONS);
	FILE *fp;
    fp = fopen(fileName.c_str(), "w");

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            fprintf(fp, "%lf, ", head[i * ROWS + j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}
