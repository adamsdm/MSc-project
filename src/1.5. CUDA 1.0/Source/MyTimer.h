#include <iostream>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

namespace MyTimer {
	std::chrono::high_resolution_clock::time_point getTime();
	double getDeltaTimeMS(std::chrono::high_resolution_clock::time_point t1, std::chrono::high_resolution_clock::time_point t2);

	void testTimer();
}