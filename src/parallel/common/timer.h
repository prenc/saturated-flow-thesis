#ifndef SATURATED_FLOW_THESIS_TIMER_H
#define SATURATED_FLOW_THESIS_TIMER_H

#include <ctime>
#include <chrono>

class Timer
{
public:
    void start();

    void stop();

    double elapsedMilliseconds();

    double elapsedNanoseconds();

private:
    std::chrono::time_point<std::chrono::system_clock> startTime;
    std::chrono::time_point<std::chrono::system_clock> endTime;
};


#endif //SATURATED_FLOW_THESIS_TIMER_H
