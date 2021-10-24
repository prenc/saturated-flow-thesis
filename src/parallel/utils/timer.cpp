#include <chrono>
#include "timer.h"

void Timer::start()
{
    startTime = std::chrono::system_clock::now();
}

void Timer::stop()
{
    endTime = std::chrono::system_clock::now();
    elapsedMilliseconds=std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    elapsedNanoseconds=std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
}