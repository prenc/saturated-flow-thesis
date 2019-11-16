#include <OpenCAL/cal2D.h>
#include <OpenCAL/cal2DIO.h>
#include <OpenCAL/cal2DRun.h>
#include <OpenCAL/cal2DBuffer.h>
#include <OpenCAL/cal2DReduction.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <string.h>

#define CA_SIZE 100
#define ROWS CA_SIZE
#define COLS CA_SIZE
#define LAYERS 1
#define CELL_SIZE_X 10
#define CELL_SIZE_Y 10
#define SPESSORE 50

#define Syinitial 0.1
#define Kinitial  0.0000125

#define headFixed 50
#define headCalculated 50

#define SIMULATION_ITERATIONS 1000

#define NUMBER_OF_WELLS 1
int rowMonitoringWell[ NUMBER_OF_WELLS] = {ROWS / 2};
int colMonitoringWell[ NUMBER_OF_WELLS] = {COLS / 2};
double monitoringWellQW[ NUMBER_OF_WELLS] = {0.001};


