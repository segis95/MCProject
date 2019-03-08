#include "utils.h"
#include <iostream>
#include <math.h>
#include <chrono>
#include <ctime>


std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(0.0, 1.0);
std::normal_distribution<double> gauss(0.0, 1.0);

int main()
{
	
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	generator.seed(seed);
	std::clock_t start;
	double duration;

	/*


	int num_sim = 1000000;

	double steps[] = { 0.05, 0.1, 0.125, 0.2, 0.25 };

	std::cout << "Number of simulations for caplets: " << num_sim << "\n";
	std::cout << "******ALGORITHM 1********\n";
	for (int j = 0; j < 5; j++) {
		start = std::clock();
		double s = 0.0;
		for (int i = 0; i < num_sim; i++) {
			s += one_simulation(0.13, 0.01, 0.28, 0.25, steps[j], 9, 1);
		}
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "step: " << steps[j] << ", result: " << s / num_sim << ", duration: " << duration << "s\n";
	}

	std::cout << "******ALGORITHM 2********\n";
	for (int j = 0; j < 5; j++) {
		start = std::clock();
		double s = 0.0;
		for (int i = 0; i < num_sim; i++) {
			s += one_simulation(0.13, 0.01, 0.28, 0.25, steps[j], 9, 2);
		}
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "step: " << steps[j] << ", result: " << s / num_sim << ", duration: " << duration << "s\n";
	}

	*/
	//std::cout << classical_method(5, 0.13, 0.01, 0.28, 0.25, 0.05, 9, 1);

	std::cout << "******CLASSICAL METHODS********\n";
	
	double s = 0.0;
	int N = 10000; // number of simulations
	double c;
	int deg = 14; // 2^deg + 1 point for brownian motion
	//*
	start = std::clock();

	std::cout << "******Number of simulations: " << N << "*****\n";
	std::cout << "******Number of points for brownian: " << deg << "*****\n";
	std::cout << "****CLASSICAL METHOD WITH PONDERATION****\n";

	for (int i = 0; i < N; i++) {
		c = classical_method(deg, 0.13, 0.01, 0.28, 0.25, 0.01, 9, 0);
		s += c ;
	}
	
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "result: " << s / N << '\n';
	std::cout << "time elapsed " << duration << "s\n\n";

	std::cout << "CLASSICAL METHOD WITHOUT PONDERATION\n";
	start = std::clock();
	s = 0.0;
	for (int i = 0; i < N; i++) {
		c = classical_method(deg, 0.13, 0.01, 0.28, 0.25, 0.01, 9, 1);
		s += c;
	}
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "result: " << s / N << '\n';
	std::cout << "time elapsed " << duration << "s\n\n";
	

}