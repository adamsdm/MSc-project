#include <iostream>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

namespace MyTimer {
	/**
	* Get current time point
	*/
	std::chrono::high_resolution_clock::time_point getTime();

	/**
	* Get delta time between two time points (ms), t1 < t2
	* @param t1		first time point
	* @param t2		second time point
	*/
	double getDeltaTimeMS(std::chrono::high_resolution_clock::time_point t1, std::chrono::high_resolution_clock::time_point t2);

}