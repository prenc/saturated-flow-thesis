#include "timer.h"

void startTimer(struct Timer *timer) {
	gettimeofday(&(timer->startTime), NULL);
}

void endTimer(struct Timer *timer) {
	gettimeofday(&(timer->endTime), NULL);
}

double getElapsedTime(struct Timer timer){
	unsigned long long seconds = timer.endTime.tv_sec - timer.startTime.tv_sec;
	unsigned long long milliseconds = (timer.endTime.tv_usec - timer.startTime.tv_usec) / 1000;
	unsigned long long totalMilliseconds = 1000 * seconds + milliseconds;
	return totalMilliseconds;
}
