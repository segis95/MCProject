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
	double s, mean, var, sim, confidence;
	
	int num_sim = 1000000;//number of simulations - change here to decrease the execution time

	double steps[] = { 0.05, 0.1, 0.125, 0.2, 0.25 };
	std::cout << "Attention! Execution of the entire of this code may take upto 30 minutes! \n";
	std::cout << "if you want to decrease the execution time please change the values of parameters num_sim and N in main.cpp\n\n";
	std::cout << "******RANDOM WALK APPROACH********\n\n";
	
	std::cout << "Number of simulations for caplets: " << num_sim << "\n\n";
	std::cout << "******ALGORITHM 1********\n";
	var = 0.0;
	for (int j = 0; j < 5; j++) {
		start = std::clock();
		s = 0.0;
		for (int i = 0; i < num_sim; i++) {
			sim = one_simulation(0.13, 0.01, 0.28, 0.25, steps[j], 9, 1);
			s += sim;
			var += sim * sim;
		}
		mean = s / num_sim;
		var = (var / num_sim - mean * mean);
		confidence = 1.96 * sqrt(var/num_sim);

		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "step: " << steps[j] << ", result: " << mean << ", IC(+-): "<< confidence << "duration: " << duration << "s\n";
	}
	std::cout << "\n\n";
	std::cout << "******ALGORITHM 2********\n";
	var = 0.0;
	for (int j = 0; j < 5; j++) {
		start = std::clock();
		s = 0.0;
		for (int i = 0; i < num_sim; i++) {
			sim = one_simulation(0.13, 0.01, 0.28, 0.25, steps[j], 9, 2);
			s += sim;
			var += sim * sim;
		}

		mean = s / num_sim;
		var = (var / num_sim - mean * mean);
		confidence = 1.96 * sqrt(var / num_sim);

		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "step: " << steps[j] << ", result: " << mean << ", IC(+-): " << confidence << ", duration: " << duration << "s\n";
	}

	
	std::cout << "\n\n\n\n";
	std::cout << "******CLASSICAL METHODS********\n\n";
	
	int N = 10000; // number of simulations - change here to decrease the execution time
	int deg = 15; // 2^deg + 1 points for brownian motion
	//*
	start = std::clock();

	std::cout << "******Number of simulations: " << N << "*****\n\n";
	std::cout << "******Number of points for brownian: 2^" << deg << "*****\n\n";
	std::cout << "******CLASSICAL METHOD WITH PONDERATION****\n";

	var = 0.0;
	s = 0.0;
	for (int i = 0; i < N; i++) {
		sim = classical_method(deg, 0.13, 0.01, 0.28, 0.25, 0.01, 9, 0);
		s += sim ;
		var += sim * sim;
	}
	
	mean = s / N;
	var = (var / N - mean * mean);
	confidence = 1.96 * sqrt(var / N);

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "result: " << mean << ", IC(+-): " << confidence << '\n';
	std::cout << "time elapsed " << duration << "s\n\n";

	std::cout << "CLASSICAL METHOD WITHOUT PONDERATION\n";
	start = std::clock();
	s = 0.0;
	var = 0.0;
	for (int i = 0; i < N; i++) {
		sim = classical_method(deg, 0.13, 0.01, 0.28, 0.25, 0.01, 9, 1);
		s += sim;
		var += sim * sim;
	}

	mean = s / N;
	var = (var / N - mean * mean);
	confidence = 1.96 * sqrt(var / N);

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "result: " << mean <<", IC(+-): "<< confidence <<'\n';
	std::cout << "time elapsed " << duration << "s\n\n";
	

}