#include "file_helper.h"

string clip_filename(string fullname){
    string name_without_ext = fullname;

    size_t last_slash_idx = name_without_ext.find_last_of("/");
    if (last_slash_idx != string::npos) {
        name_without_ext = name_without_ext.substr(last_slash_idx + 1);
    }

    size_t last_dot_idx = name_without_ext.find_last_of(".");
    if (last_dot_idx != string::npos) {
        name_without_ext = name_without_ext.substr(0, last_dot_idx);
    }

    return name_without_ext;
}

void writeHeadsToFile(double *head, string test_name) {
    create_output_dir(OUTPUT_PATH);

    string filename = OUTPUT_PATH + clip_filename(test_name) + "_heads.csv";

    write_to_file(head, filename);
}

void write_river_heads_to_file(double *head, double river_head, int day) {
    string output_path = "./output/river/";
    create_output_dir(output_path);
    string filename = output_path + to_string(day);
    write_to_file(head, filename);
}

void create_output_dir(string path) {
    if (stat(path.c_str(), &st) == -1) {
        mkdir(path.c_str(), 0770);
    }
}

void write_to_file(double *head, string filename) {
    FILE *fp = fopen(filename.c_str(), "w");

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if(TRANSPOSE_OUTPUT == 1){
                fprintf(fp, "%.15lf, ", head[j * ROWS + i]);
            }else{
                fprintf(fp, "%.15lf, ", head[i * ROWS + j]);
            }
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}
