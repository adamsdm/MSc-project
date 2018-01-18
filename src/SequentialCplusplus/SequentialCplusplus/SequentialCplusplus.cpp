// SequentialCplusplus.cpp : Defines the entry point for the console application.
//

#include <iostream>

#define N_elements	102400000

void add(int *vector1, int *vector2, int *vector3) {
	int i;
	for (i = 0; i < N_elements; i++){
		vector3[i] = vector1[i] + vector2[i];
	}
}

int main(){

	/*
	int *vector1 = (int *)malloc(N_elements * sizeof(int));
	int *vector2 = (int *)malloc(N_elements * sizeof(int));
	int *vector3 = (int *)malloc(N_elements * sizeof(int));
	*/

	int *vector1 = new int[N_elements];
	int *vector2 = new int[N_elements];
	int *vector3 = new int[N_elements];


	int i;
	for (i = 0; i < N_elements; i++) {
		vector1[i] = i;
		vector2[i] = 2*i;
	}


	add(vector1, vector2, vector3);

	// Control result
	for (i = 0; i < N_elements; i++){
		if (vector3[i] != vector1[i] + vector2[i]) {
			delete[] vector1; delete[] vector2; delete[] vector3;
			return EXIT_FAILURE;
		}
			
	}

	std::cout << "Success!" << std::endl;
	delete[] vector1; delete[] vector2; delete[] vector3;
    return EXIT_SUCCESS;
}

