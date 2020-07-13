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
	return timer.endTime.tv_usec - timer.startTime.tv_usec;
}

