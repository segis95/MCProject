#pragma once
#include <random>
#define M_SQRT1_2 sqrt(0.5)
#define PI 3.141592653589793238463

double d_p(double x, double v);
double d_m(double x, double v);
double cdf(double value);
double pdf(double x);
double d_prime(double x, double v);
double dF_dL(double L_i, double K, double H, double v_i);
double libor_price(double L_i, double K, double H, double v_i);
double bern(double p);
double pos_part(double x);
double one_simulation(double L_i, double K, double H, double sigma, double h, double i, int mode);



extern std::default_random_engine generator;
extern std::uniform_real_distribution<double> distribution;

