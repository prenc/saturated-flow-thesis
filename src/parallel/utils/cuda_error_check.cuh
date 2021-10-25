#ifndef SATURATED_FLOW_ERROR
#define SATURATED_FLOW_ERROR

#include <cstdio>

#define ERROR_CHECK(err) __cudaSafeCall(err, __FILE__, __LINE__)

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));

        fprintf(stdout, "cudaSafeCall() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
    return;
}

#endif //SATURATED_FLOW_ERROR