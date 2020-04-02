//
// Created by prenc on 11/10/19.
//


#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//MODEL PARAMS

#define CA_SIZE 1000
#define SIMULATION_ITERATIONS 1000
#define BLOCK_SIZE 16
#define WRITE_OUTPUT_TO_FILE 0

#define ROWS CA_SIZE
#define COLS CA_SIZE

#define CELL_SIZE_X 0.1
#define CELL_SIZE_Y 0.1

#define AREA CELL_SIZE_X*CELL_SIZE_Y

#define THICKNESS 50
#define Syinitial 0.1
#define Kinitial  0.0000125

#define headFixed 50
#define headCalculated 50

#define DELTA_T 0.4
#define KERNEL_LOOP_SIZE 50

//multiple wells
#define NUMBER_OF_WELLS 1
int wellsRows[NUMBER_OF_WELLS] = {ROWS / 2};
int wellsCols[NUMBER_OF_WELLS] = {COLS / 2};
double wellsQW[NUMBER_OF_WELLS] = {0.001};

struct CA {
    double *head;
    double *Sy;
    double *K;
    double *Source;
};
