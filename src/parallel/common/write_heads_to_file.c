#include <iostream>
#include <stdlib.h>
#include <stdio.h>

void write_heads_to_file(double *head) {
    FILE *fp;
    fp = fopen("heads_ca.txt", "w");

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            fprintf(fp, "%lf, ", head[i * ROWS + j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}
