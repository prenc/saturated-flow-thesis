#ifndef SATURATED_FLOW_THESIS_TIMER_H
#define SATURATED_FLOW_THESIS_TIMER_H

#include <cstddef>
#include <sys/time.h>

struct Timer{
	struct timeval startTime;
	struct timeval endTime;
	struct timeval elapsedTime;
};

void startTimer(struct Timer *timer);
void endTimer(struct Timer *timer);
double getElapsedTime(struct Timer timer);

#endif //SATURATED_FLOW_THESIS_TIMER_H
