#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//MODEL PARAMS

#define CA_SIZE 1000
#define SIMULATION_ITERATIONS 4000
#define BLOCK_SIZE 32

#define WRITE_OUTPUT_TO_FILE 0
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
#define KERNEL_LOOP_SIZE 20

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
#define WELLS_DIM 1
#define NUMBER_OF_WELLS WELLS_DIM*WELLS_DIM
#define WELLS_Y ROWS / 2
#define WELLS_X COLS / 2
#define WELLS_QW 0.001

int wellsRows[] = { WELLS_Y };
int wellsCols[] = { WELLS_X };
double wellsQW[] = { WELLS_QW };

struct CA {
    double *head;
    double *Sy;
    double *K;
    double *Source;
};
