#include "MyTimer.h"


std::chrono::high_resolution_clock::time_point MyTimer::getTime(){
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	return t1;
}

double MyTimer::getDeltaTimeMS(
	std::chrono::high_resolution_clock::time_point t1, 
	std::chrono::high_resolution_clock::time_point t2)
{
	std::chrono::duration<double, std::milli> time_span = t2 - t1;
	return time_span.count();
}
