#include <stdio.h>

#define ROWS 100
#define COLS 100
#define CELL_SIZE_X 10
#define CELL_SIZE_Y 10
#define THICKNESS 50

#define Syinitial 0.1
#define Kinitial  0.0000125

#define headFixed 50
#define headCalculated 50

#define SIMULATION_ITERATIONS 1000

double delta_t_ = 4000;
double qw = 0.001;

int posSy = ROWS / 2;
int posSx = COLS / 2;

struct {
    double head[ROWS][COLS];
    double Sy[ROWS][COLS];
    double K[ROWS][COLS];
    double Source[ROWS][COLS];
} ca;

void init_ca();

void simulation_step();

void transition_function(int i, int j);

void count_q(double *pDouble, int i, int j, int i1, int j1);

void write_heads_to_file();

int main() {
    init_ca();

    for (int i = 0; i < SIMULATION_ITERATIONS; i++) {
        simulation_step();
    }

    write_heads_to_file();

    return 0;
}

void write_heads_to_file() {
    FILE *fp;
    fp = fopen("heads_no_rw_cas.txt", "w");

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            fprintf(fp, "%lf, ", ca.head[i][j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

void simulation_step() {
    for (int i = 0; i < ROWS; i++)
        if (i != 0 && i != ROWS - 1)
            for (int j = 0; j < COLS; j++) {
                transition_function(i, j);
            }
}

void transition_function(int i, int j) {

    double Q = 0;

    if (i - 1 >= 0) {
        count_q(&Q, i, j, i - 1, j);
    }
    if (j - 1 >= 0) {
        count_q(&Q, i, j, i, j - 1);
    }
    if (i + 1 < ROWS) {
        count_q(&Q, i, j, i + 1, j);
    }
    if (j + 1 < COLS) {
        count_q(&Q, i, j, i, j + 1);
    }

    Q -= ca.Source[i][j];
    double area = CELL_SIZE_X * CELL_SIZE_Y;
    double ht1 = (Q * delta_t_) / (area * ca.Sy[i][j]);

    ca.head[i][j] += ht1;

}

void count_q(double *pDouble, int i, int j, int i1, int j1) {
    double diff_head = ca.head[i1][j1] - ca.head[i][j];
    double tmp_t = ca.K[i1][j1] * THICKNESS;

    *pDouble += diff_head * tmp_t;
}


void init_ca() {
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++) {
            ca.head[i][j] = headFixed;
            if (j == COLS - 1) {
                ca.head[i][j] = headCalculated;
            }
            ca.Sy[i][j] = Syinitial;
            ca.K[i][j] = Kinitial;
            ca.Source[i][j] = 0;
        }

    ca.Source[posSy][posSx] = qw;
}