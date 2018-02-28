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

void MyTimer::testTimer(){
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	std::cout << "printing out 1000 stars...\n";
	for (int i = 0; i<1000; ++i) std::cout << "*";
	std::cout << std::endl;

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> time_span = t2 - t1;

	std::cout << "It took me " << time_span.count() << " ms.";
	std::cout << std::endl;

}
