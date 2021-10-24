#include <filesystem>
#include "file_helper.h"

void saveHeadsInFile(double *&head, char *test_name)
{
    std::filesystem::path testName(test_name);
    std::filesystem::path testResultPath(OUTPUT_PATH);
    testResultPath /= "heads_";
    testResultPath += testName.stem().string();
    testResultPath += ".csv";
    writeHeads(head, testResultPath.string());
}

void saveRiverHeadsInFile(double *&head, double &river_head, int &day)
{
    std::filesystem::path riverResultPath(OUTPUT_PATH);
    riverResultPath /= "river";
    riverResultPath /= std::to_string(day);
    writeHeads(head, riverResultPath.string());
}

void writeHeads(double *&heads, const std::string &file_path)
{
    std::filesystem::path filePath(file_path);
    std::filesystem::create_directories(filePath.parent_path());
    FILE *fp = fopen(filePath.c_str(), "w");

    for (int i = 0; i < ROWS; i++)
    {
        for (int j = 0; j < COLS; j++)
        {
            fprintf(fp, "%.15lf, ", heads[i * ROWS + j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}
