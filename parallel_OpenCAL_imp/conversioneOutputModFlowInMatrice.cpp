#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#define ROWS 100
#define COLS 100

#define STRLEN 256
void calfLoadMatrix2Dr(double* M, int rows, int columns, FILE* f)
{
  char str[STRLEN];
  int i, j;

  for (i=0; i<rows; i++)
    for (j=0; j<columns; j++){
      fscanf(f, "%s", str);
      M[i*columns+j] = atof(str);
      //calSetMatrixElement(M, columns, i, j, atoi(str));
    }
}

bool calLoadMatrix2Dr(double* M, int rows, int columns, char* path)
{
  FILE *f = NULL;
  f = fopen(path, "r");

  if ( !f )
    return false;

  calfLoadMatrix2Dr(M, rows, columns, f);

  fclose(f);

  return true;
}
//FILE * monitorWellFile;
const int numberofWell = 5;
FILE * monitorWellFiles[5]; //= malloc(sizeof(FILE)*numberofWell);


const char* name = "monitorWellModFlow";
const char* extension = ".txt";

//double convergence = ((CELL_SIZE_X*CELL_SIZE_Y)*Syinitial)/(Kinitial*4);

int rowMonitoringWell[5] = {49,49,49, 40,37};
int colMonitoringWell[5] = {34,29,19, 40,37}; 


int main(){
    // run     sed '/HEAD/d' file.fhd    to delete HEAD of each iteration

    for(int i = 0; i < numberofWell; i++){
        char* name_with_extension;
        
        name_with_extension = (char*) malloc(strlen(name)+1); /* make space for the new string (should check the return value ...) */
        
        strcpy(name_with_extension, name); /* copy name into the new var */
        
        char char_arr [100];
        //printf("%i\n", colMonitoringWell[i]);
        sprintf(char_arr, "%d", colMonitoringWell[i]);
        strcat(name_with_extension, char_arr);
        strcat(name_with_extension, extension); /* add the extension */ 
       //printf("%s\n", name_with_extension);

        monitorWellFiles[i] =  fopen(name_with_extension, "w");
        fclose(monitorWellFiles[i]);
        monitorWellFiles[i] =  fopen(name_with_extension, "a");    
    }


    double * lastModFlow;
    int rowsModFlow = 259000;
    int colsModFlow = 10;
    lastModFlow = (double*) malloc(sizeof(double)*(rowsModFlow*colsModFlow));

    calLoadMatrix2Dr(lastModFlow, rowsModFlow, colsModFlow, "./ravazzani2.fhd");

    int numberofMatrix = 259; // this variables identies the number of computational steps example (12*86400)/4000 = 259

    double* lastModFlowCorretta = (double*) malloc(sizeof(double)*(ROWS*COLS*numberofMatrix));

    int cont = 0;
    for(int i = 0; i< rowsModFlow;i++){
       for(int j = 0; j< colsModFlow;j++){
            lastModFlowCorretta[ cont ] =  lastModFlow[ i*colsModFlow+j ];
            cont++;
            //printf("%d\t%d\t%f\n",i+1,j+1,lastModFlow[i*colsModFlow+j]);
       }   
    //     //printf(" \n");
    }

    // for(int i = 0; i< ROWS;i++){
    //    for(int j = 0; j< COLS;j++)
    //         printf("%d\t%d\t%f\n",j+1,i+1,lastModFlowCorretta[i*COLS+j]);
    //     //printf(" \n");
    // }

    // for(int k = 0;k< numberofMatrix; k++){
    //     for(int i = 0; i< ROWS;i++){
    //         for(int j = 0; j< COLS;j++){
    //             printf("%f ",lastModFlowCorretta[k*ROWS*COLS + (i*COLS+j)]);
    //         }
    //         printf("\n");
    //     }
    // }

    int delta_t_ = 4000;
    int numberofdays=1;
    for(int k = 0;k< numberofMatrix;k++){
        double time_ =delta_t_*k;
        int days = time_/86400;

        if((numberofdays)*86400 <= time_)
        {
            for(int i = 0; i < numberofWell; i++){
                fprintf(monitorWellFiles[i], "%d %f \n", days, 50-lastModFlowCorretta[k*ROWS*COLS + (rowMonitoringWell[i]*COLS+colMonitoringWell[i])]);
            }
            numberofdays++;
        }
        
    }

    int k = numberofMatrix-1;
    for(int i = 0; i < numberofWell; i++){
                fprintf(monitorWellFiles[i], "%d %f \n", 12, 50-lastModFlowCorretta[k*ROWS*COLS + (rowMonitoringWell[i]*COLS+colMonitoringWell[i])]);
    }


    for(int i = 0; i < numberofWell; i++){
       fclose(monitorWellFiles[i]); 
    }


    return 0;
}