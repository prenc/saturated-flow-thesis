//
// Created by pecatoma on 13.07.2020.
//
#include <cstddef>
#include <sys/time.h>

struct Timer{
	struct timeval startTime;
	struct timeval endTime;
	struct timeval elapsedTime;
};

void startTimer(Timer *timer) {
	gettimeofday(&(timer->startTime), NULL);
}

void endTimer(Timer *timer) {
	gettimeofday(&(timer->endTime), NULL);
}

double getElapsedTime(Timer timer){
	unsigned long long seconds = timer.endTime.tv_sec - timer.startTime.tv_sec;
	unsigned long long milliseconds = (timer.endTime.tv_usec - timer.startTime.tv_usec) / 1000;
	unsigned long long totalMilliseconds = 1000 * seconds + milliseconds;
	return totalMilliseconds;
}

