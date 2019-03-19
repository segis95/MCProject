#pragma once
#include "linalg.h"
#include "stdafx.h"
#include "optimization.h"
#include <random>
#include <fstream>


// structure defined for passing arguments to objective function used by optimizer
struct args{
    int N;
    double R_up;
    double delta;
    alglib::real_1d_array log_L;
    double (*get) (const alglib::real_1d_array& , int, double, double);
};

typedef struct outCome{
    int projection_num;
    double projection_time;
    double timeStamp;
    alglib::real_1d_array logL;
} outCome;

// get correlation matrix rho
void init_Rho(alglib::real_2d_array& rho, double beta, double* T,int N);

// print matrix
void print_Matrix(alglib::real_2d_array Mat, int M, int N);

// cholesky of Mat, resualt stored in U
void get_U(alglib::real_2d_array& U, double beta, double* T, int N);

// calculate R_swap
double R_Swap(alglib::real_1d_array& log_L, double delta, int N, bool isLog=true);

// generate one random variable following bernouill law
double get_Bern(double p);

// get norm , q specifies the order of the norm
double get_Norm(alglib::real_1d_array& proj_logL, alglib::real_1d_array& log_L, int N, int q = 2);

// generate boudary zone probability by (49)
double get_p(double norm, int N, double h,double sigma);

// log_L jumps back into the domaine G
void jump_Back(alglib::real_1d_array& log_L, alglib::real_1d_array& proj_logL, int N, double h, double sigma);

// generate randomStep vector for one step simulatioin
void get_RandomStep(alglib::real_1d_array& randomStep, double p, int N);

// do one update 
void one_Simulation(alglib::real_1d_array& log_L, alglib::real_1d_array& L_, alglib::real_2d_array& U, alglib::real_2d_array& Rho, double delta, double sigma, double h, int N);

// auxilliary function for first test
double get_MaxlogL(alglib::real_1d_array& log_L, int N);
// test follows equation (43) in article
bool first_Test(alglib::real_1d_array& log_L, int N, double sigma, double h, double R_up);

// auxilliary functions for projection
double get_log_L0(const alglib::real_1d_array& log_L, int N, double sigma, double R_up);
void fonction_fvec(const alglib::real_1d_array& x, alglib::real_1d_array& fi, void *ptr);
alglib::real_1d_array projection(alglib::real_1d_array& proj_X, void (*fvec)(const alglib::real_1d_array &x, alglib::real_1d_array &fi, void *ptr), void *ptr);

// test follows equation (44)
bool second_Test(alglib::real_1d_array& log_L, alglib::real_1d_array& L, int N, double sigma, double h, double delta, double R_up);

// simulation
// void simulation(alglib::real_1d_array& log_L, alglib::real_2d_array& U, alglib::real_2d_array& Rho, double delta, double sigma, double h, int N, double T0,int itr, bool is_Recording);
// simulation with 2 test and projection
outCome real_simulation(alglib::real_1d_array& log_L, alglib::real_2d_array& U, alglib::real_2d_array& Rho, double R_up, double delta, double sigma, double h, int N, double T0, int itr, std::ofstream& record, bool is_Recording=true);

// libor to zero coupon
alglib::real_1d_array get_zeroCoupon(alglib::real_1d_array& log_L, int N, double delta);
// calculate the price
double get_Price(alglib::real_1d_array& zeroCoupon, double Rswap, double K, double delta, int N);

// monte carlo
void monte_carlo(int num_simulation, alglib::real_1d_array& log_L, alglib::real_2d_array& U, alglib::real_2d_array& Rho, double R_up, double delta, double sigma, double h, int N, double T0, double actualisation, double K, std::ofstream& record, bool is_Recording=true);

// 
void print_logL(alglib::real_1d_array& log_L, int N);
void record_logL(alglib::real_1d_array& log_L, int N, double timeStamp, std::ofstream& record);


// extern std::default_random_engine generator;
extern std::random_device generator;
extern std::uniform_real_distribution<double> distribution;
