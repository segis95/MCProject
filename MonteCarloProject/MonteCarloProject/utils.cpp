#include <iostream>
#include <random>
#include <math.h>
#include <chrono>
#include <ctime>
#include "utils.h"



double d_p(double x, double v) {
	return (log(x) + v * v / 2) / v;
}

double d_m(double x, double v) {
	return (log(x) - v * v / 2) / v;
}

double cdf(double value) {
	return 0.5 * erfc(-value * M_SQRT1_2);
}

double pdf(double x) {
	return exp(-x * x / 2) / sqrt(2 * PI);
}

double d_prime(double x, double v) {
	return 1.0 / (x * v);
}

double dF_dL(double L_i, double K, double H, double v_i) {
	return cdf(d_p(L_i / K, v_i)) - cdf(d_p(L_i / H, v_i)) + \
		L_i * (pdf(d_p(L_i / K, v_i)) * d_prime(L_i / K, v_i) / K - \
			pdf(d_p(L_i / H, v_i)) * d_prime(L_i / H, v_i) / H) - \
		K * (pdf(d_m(L_i / K, v_i)) * d_prime(L_i / K, v_i) / K - \
			pdf(d_m(L_i / H, v_i)) * d_prime(L_i / H, v_i) / H) - \
		H * (-pdf(d_p(H * H / K / L_i, v_i)) * d_prime(H * H / K / L_i, v_i) * H * H / K / L_i / L_i + \
			pdf(d_p(H / L_i, v_i)) * H / L_i / L_i) * d_prime(H / L_i, v_i) + \
		K / H * (cdf(d_m(H * H / K / L_i, v_i)) - cdf(d_m(H / L_i, v_i))) + \
		K * L_i / H * (-pdf(d_m(H * H / K / L_i, v_i)) * H * H / K / L_i / L_i * d_prime(H * H / K / L_i, v_i) + \
			pdf(d_m(H / L_i, v_i)) * H / L_i / L_i * d_prime(H / L_i, v_i));

}

double libor_price(double L_i, double K, double H, double v_i) {
	return L_i * (cdf(d_p(L_i / K, v_i)) - cdf(d_p(L_i / H, v_i))) - \
		K * (cdf(d_m(L_i / K, v_i)) - cdf(d_m(L_i / H, v_i))) - \
		H * (cdf(d_p(H * H / K / L_i, v_i)) - cdf(d_p(H / L_i, v_i))) + \
		K * L_i / H * (cdf(d_m(H * H / K / L_i, v_i)) - cdf(d_m(H / L_i, v_i)));
}


double bern(double p) {
	return distribution(generator) < p ? -1.0 : 1.0;
}

double pos_part(double x) {
	return x > 0.0 ? x : 0.0;
}


double one_simulation(double L_i, double K, double H, double sigma, double h, double i, int mode) {

	int M = int(i / h);
	int k = 0;
	double lnL = log(L_i);
	double Z = 0.0;
	double sqh = sqrt(h);
	double v_i = sigma * sqrt(i);

	double lnH = log(H);
	double lambda_sqh = -0.5 * sigma * sigma * h + sigma * sqh;

	double value_to_check = lnH - lambda_sqh;
	double eta;

	while (k < M) {

		if (lnL >= value_to_check) {
			if (mode == 2)
				return Z;
			else {
				if (bern(lambda_sqh / (lnH - lnL + lambda_sqh)) < 0.0)
					return Z;
				else
					lnL = lnL - lambda_sqh;
			}
		}

		eta = bern(0.5);
		Z = Z - sigma * dF_dL(exp(lnL), K, H, v_i) * sqh * eta;
		lnL = lnL - 0.5 * sigma * sigma * h + sigma * sqh * eta;
		k += 1;
	}

	return pos_part(exp(lnL) - K) + Z;

}
double logn(double x, double sigma, double t) {
	return exp(-sigma * sigma * t / 2.0 + sigma * x);
}

double classical_method(int N, double L_i, double K, double H, double sigma, double h, double i, int mode) {
	
	int lgth = pow(2, N);
	std::vector<double> W(lgth + 1);
	int k;
	int step;

	double tc = sqrt(i);

	W[0] = 0.0;
	W[lgth] = gauss(generator) * tc;
	
	double Z;
	int level = lgth;

	for (int j = 1; j < N + 1; j++) {
		level /= 2;
		step = level * 2;
		for (k = level; k < lgth; k = k + step) {
			Z = gauss(generator) * (sqrt(double(level) / lgth / 2.0) * tc);
			W[k] = (W[k - level] + W[k + level]) / 2.0 + Z;
			//std::cout << Z << ' ' << W[k - level] << ' ' << W[k] <<' '<<W[k + level] << '\n';

		}
	}

	//return W[1];
	double max = 0.0;
	double delta = 1.0 / lgth; 
	
	double multiplier = 1.0;
	for (int j = lgth; j >= 0; j--) {
		W[j] = L_i * logn(W[j], sigma, i * delta * j);
		
		if (W[j] > max) {
			max = W[j];
			//std::cout << max << '\n'; 
		}
		if (j < lgth) {
			multiplier *= (1.0 - exp(-2.0 * double(lgth) / (i * sigma * sigma) * \
				(W[j] - H) * (W[j + 1] - H))) ;
		}
	}
	//return W[lgth];
	//std::cout << multiplier << '\n';
	if (max < H) {
		if (mode == 0)
			return W[lgth] - K > 0.0 ? (W[lgth] - K) * multiplier : 0.0;
		else
			return W[lgth] - K > 0.0 ? (W[lgth] - K) : 0.0;
	}
	else
		//std::cout << 1;
		return 0.0;
}
