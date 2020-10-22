#include <chrono>
#include "timer.h"

void Timer::start()
{
    startTime = std::chrono::system_clock::now();
}

void Timer::stop()
{
    endTime = std::chrono::system_clock::now();
}

double Timer::elapsedMilliseconds()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
}

double Timer::elapsedNanoseconds()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
}
