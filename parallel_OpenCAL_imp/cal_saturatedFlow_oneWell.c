#include "../parallel_CUDA_imp/src/params.h"

#include <OpenCAL/cal2D.h>
#include <OpenCAL/cal2DIO.h>
#include <OpenCAL/cal2DRun.h>

#define ROWS CA_SIZE
#define COLS CA_SIZE

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


struct CALModel2D* satured;
struct CALSubstate2Dr* head;
struct CALSubstate2Dr* K;
struct CALSubstate2Dr* Sy;
struct CALSubstate2Dr* Sorgente;
struct CALSubstate2Dr* convergence;
struct CALSubstate2Dr* Mod;

struct CALRun2D* satured_simulation;

float delta_t_ = 4000;

int numberofdays =1;
void saturedTransitionFunctionHw(struct CALModel2D* ca)
{
    double time_ =delta_t_*satured_simulation->step;


    if((numberofdays)*86400 <= time_)
    {
        numberofdays++;
    }
}

void saturedTransitionFunction(struct CALModel2D* ca, int i, int j)
{
    if(j==0 || j ==COLS-1){

        return;
    }
    double diffHead=0.0;

    double Q=0.0;
    double tmpT = 0.0;
    for (int n=1; n<ca->sizeof_X; n++){
      if ( i == 0 && n == 1){

        diffHead = 0;
      }else if (i == ROWS -1 && n == 4){

        diffHead = 0;


      }else{

            diffHead = (calGetX2Dr(ca, head, i, j, n)- calGet2Dr(ca, head, i, j));
      }



            tmpT = calGetX2Dr(ca, K, i, j, n)*SPESSORE;

            Q += (tmpT * diffHead);
    }
    Q+=-calGet2Dr(ca, Sorgente, i, j);

    double area = CELL_SIZE_X*CELL_SIZE_Y;
    double ht1 = (Q*delta_t_)/(calGet2Dr(ca, Sy, i, j)*area);
    calSet2Dr(ca, head, i, j, ht1+calGet2Dr(ca, head, i, j));
}

void saturedInit(struct CALModel2D* ca)
{
	for (int i = 0; i < ROWS; i++)
		for (int j = 0; j < COLS; j++)
		{
			if (j == COLS - 1)
				calSet2Dr(satured, head, i, j, headCalculated);
			else
				calSet2Dr(satured, head, i, j, headFixed);
		}

	for (int i = 0; i < ROWS; i++)
		for (int j = 0; j < COLS; j++)
		{
			calSet2Dr(satured, K, i, j, Kinitial);
			calSet2Dr(satured, Sy, i, j, Syinitial);
		}

	for (int i = 0; i < ROWS; i++)
		for (int j = 0; j < COLS; j++)
		{
			calSet2Dr(satured, Sorgente, i, j, 0);
		}
	int x, y;
	double source;
	for (int i = 0; i < NUMBER_OF_WELLS; i++)
	{
		x = rowMonitoringWell[i];
		y = colMonitoringWell[i];
		source = monitoringWellQW[i];
		calSet2Dr(satured, Sorgente, x, y, source);
	}
}

int main(){
	satured = calCADef2D(ROWS, COLS, CAL_VON_NEUMANN_NEIGHBORHOOD_2D, CAL_SPACE_FLAT, CAL_NO_OPT);
	satured_simulation = calRunDef2D(satured, 1, SIMULATION_ITERATIONS, CAL_UPDATE_IMPLICIT);

	head = calAddSubstate2Dr(satured);
    K = calAddSubstate2Dr(satured);
    Sy = calAddSubstate2Dr(satured);
    Sorgente = calAddSubstate2Dr(satured);
    convergence = calAddSubstate2Dr(satured);

	calAddElementaryProcess2D(satured, saturedTransitionFunction);
    calRunAddSteeringFunc2D(satured_simulation, saturedTransitionFunctionHw);


    calRunAddInitFunc2D(satured_simulation, saturedInit);
    calRunInitSimulation2D(satured_simulation);

	  calRun2D(satured_simulation);

    calSaveSubstate2Dr(satured, head, "./satured_head_LAST.txt"); // write final heads level to file

	calRunFinalize2D(satured_simulation);
	calFinalize2D(satured);
}