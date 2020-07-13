#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//MODEL PARAMS

#define CA_SIZE 4000
#define SIMULATION_ITERATIONS 1000
#define BLOCK_SIZE 32

#define WRITE_OUTPUT_TO_FILE 1
#define TRANSPOSE_OUTPUT 1

#define ROWS CA_SIZE
#define COLS CA_SIZE

#define CELL_SIZE_X 1
#define CELL_SIZE_Y 1

#define AREA CELL_SIZE_X*CELL_SIZE_Y

#define THICKNESS 50
#define Syinitial 0.1
#define Kinitial  0.0000125

#define headFixed 50
#define headCalculated 50

#define DELTA_T 40
#define KERNEL_LOOP_SIZE 50

#define WRITE_STATISTICS_TO_FILE 1
#define STATISTICS_WRITE_FREQ 10
// RIVER
#define KSB  0.00001 // streambed hydraulic conductivity
#define M  0.5 // river bed thickness
#define W  5 // river width

#define RIVER_BOTTOM  46.5
#define RIVER_POSITION 24

#define SIMULATED_DAYS_NUMBER 10
double river_heads[SIMULATED_DAYS_NUMBER] = { 49, 49, 49, 49, 49, 51, 51, 51, 51, 51 };

#define SECONDS_IN_DAY  86400
#define SIMULATION_STEPS (SECONDS_IN_DAY * SIMULATED_DAYS_NUMBER) / DELTA_T

//multiple wells
#define NUMBER_OF_WELLS 1
int wellsRows[NUMBER_OF_WELLS] = { ROWS / 2 };
int wellsCols[NUMBER_OF_WELLS] = { COLS / 2 };
double wellsQW[NUMBER_OF_WELLS] = { 0.001 };

struct CA {
    double *head;
    double *Sy;
    double *K;
    double *Source;
};
