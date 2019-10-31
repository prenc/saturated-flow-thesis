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

struct CA {
    double head[ROWS][COLS];
    double Sy[ROWS][COLS];
    double K[ROWS][COLS];
    double Source[ROWS][COLS];
} read, write;

void init_ca();

void simulation_step();

void transition_function(int i, int j);

void count_q(double *pDouble, int i, int j, int i1, int j1);

void write_heads_to_file();

void copy_heads(struct CA *r, struct CA *w);

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
            fprintf(fp, "%lf, ", read.head[i][j]);
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
    copy_heads(&write, &read);
}

void copy_heads(struct CA *w, struct CA *r) {
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++) {
            r->head[i][j] = w->head[i][j];
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

    Q -= read.Source[i][j];
    double area = CELL_SIZE_X * CELL_SIZE_Y;
    double ht1 = (Q * delta_t_) / (area * read.Sy[i][j]);

    write.head[i][j] += ht1;
}

void count_q(double *pDouble, int i, int j, int i1, int j1) {
    double diff_head = read.head[i1][j1] - read.head[i][j];
    double tmp_t = read.K[i1][j1] * THICKNESS;

    *pDouble += diff_head * tmp_t;
}


void init_ca() {
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++) {
            read.head[i][j] = headFixed;
            if (j == COLS - 1) {
                read.head[i][j] = headCalculated;
            }
            read.Sy[i][j] = Syinitial;
            read.K[i][j] = Kinitial;
            read.Source[i][j] = 0;
        }

    read.Source[posSy][posSx] = qw;
    copy_heads(&read, &write);
}