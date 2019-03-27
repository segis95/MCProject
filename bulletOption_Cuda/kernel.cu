
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "rng.h"
#include <device_functions.h>

#include <cuda.h>

//#include <string>

#define nt 15
#define nk 6
#define L 100 //time step number
#define M 9 // payment date number
#define N 100// space step number


// read only variables in GPU defined by prof, used by local volatility
__constant__ float Tg[nt];
__constant__ float rg[nt];// interest rate, constant par morceaux
__constant__ float Kg[nk];
__constant__ float Cg[16 * (nt - 1)*(nk - 1)];

__constant__ double err_r;//; = 1e-4;

__constant__ float Kgmax;// = 250.f;
__constant__ float Kgmin;// = 20.f;
__constant__ float K;// = 100.f;
__constant__ int P1;// = 2;
__constant__ int P2;// = 8;
__constant__ float B;// = 120.f;

// we should define the maturity here 
// since the interest rate defined in rg has max time about 3 years, so we define maturity as 3 years
// so that in our simulation, all time steps has a defined interest rate
__constant__ float TT = 3.f;// SER_c: might be a name collision error with array T
__constant__ float delta_T;// = TT / L;

__constant__ float delta_X;// = (log(Kgmax) - log(Kgmin)) / N;// ATTENTION: X is log
// prefixed dates correspond to maturity T = 3
__constant__ float T[M] = { 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7 };

__device__ bool is_synchronized[L];
__device__ int synchro_count[L];
__device__ float As[M*(N - 1)];
__device__ float Bs[M*N];
__device__ float Cs[M*(N - 1)];
__device__ float Ys[M*N];
__device__ float Zs[M*N];


// variables on CPU defined by prof, used to transfer values from host to devices
float *Cgc, *Kgc, *Tgc, *rgc;

//
bool *is_synchronized_c;
int *synchro_count_c;
float *Asc, *Bsc, *Csc, *Ysc, *Zsc;

// Allocate all parameters for local volatility
void VarMalloc()
{
	// variables used for local volatility
	Kgc = (float *)calloc(nk, sizeof(float));
	Tgc = (float *)calloc(nt, sizeof(float));
	rgc = (float *)calloc(nt, sizeof(float));
	Cgc = (float *)calloc(16 * (nk - 1)*(nt - 1), sizeof(float));

	// variables used for solve_PDE kernel
	is_synchronized_c = (bool *)calloc(L, sizeof(bool));
	synchro_count_c = (int *)calloc(L, sizeof(int));
	
	Asc = (float *)calloc(M * (N - 1), sizeof(float));
	Bsc = (float *)calloc(M * N, sizeof(float));
	Csc = (float *)calloc(M * (N - 1), sizeof(float));
	Ysc = (float *)calloc(M * N, sizeof(float));
	Zsc = (float *)calloc(M * N, sizeof(float));
	
}

// Free all parameters
void FreeVar()
{
	free(Cgc);
	free(Kgc);
	free(Tgc);
	free(rgc);
	free(is_synchronized_c);
	free(synchro_count_c);
	free(Asc);
	free(Bsc);
	free(Csc);
	free(Ysc);
	free(Zsc);
}


__device__ float Max(float X, float Y) {
	return X < Y ? X : Y;
}
// Initialize all parameters
void parameters()
{
	// initialization all parameters for PDE_solver
	for (int i = 0; i < L; i++) {
		is_synchronized_c = false;
	}
	is_synchronized_c[L - 1] = true;
	for (int i = 0; i < L; i++) {
		synchro_count_c[i] = 0;
	}
	for (int i = 0; i < M*N; i++) {
		Bsc[i] = 0.f;
		Ysc[i] = 0.f;
	}
	// initialization all Z to terminal condition
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			if (j < P1 || j > P2) {
				Zsc[i + j * N] = 0.f;
			}
			else {
				double x = log(Kgmin) + i * delta_X;
				Zsc[i + j * N] = Max(exp(x) - K, 0.f);
			}
		}
	}
	for (int i = 0; i < M*(N - 1); i++) {
		Asc[i] = 0.f;
		Csc[i] = 0.f;
	}

	cudaMemcpyToSymbol(is_synchronized, is_synchronized_c, L * sizeof(bool));
	cudaMemcpyToSymbol(synchro_count, synchro_count_c, L * sizeof(int));
	cudaMemcpyToSymbol(As, Asc, M*(N - 1) * sizeof(float));
	cudaMemcpyToSymbol(Cs, Csc, M*(N - 1) * sizeof(float));
	cudaMemcpyToSymbol(Bs, Bsc, M*N * sizeof(float));
	cudaMemcpyToSymbol(Ys, Ysc, M*N * sizeof(float));
	cudaMemcpyToSymbol(Zs, Zsc, M*N * sizeof(float));


	// initialization for local volatility by prof
	Kgc[0] = 20.f;
	Kgc[1] = 70.f;
	Kgc[2] = 120.f;
	Kgc[3] = 160.f;
	Kgc[4] = 200.f;
	Kgc[5] = 250.0f;

	float d, w, m, y;
	d = 1.0f / 360.0f;
	w = 7.0f * d;
	m = 30.0f * d;
	y = 12.0f * m;

	Tgc[0] = d;
	Tgc[1] = 2.f*d;
	Tgc[2] = w;
	Tgc[3] = 2.f*w;
	Tgc[4] = m;
	Tgc[5] = 2.f*m;
	Tgc[6] = 3.f*m;
	Tgc[7] = 6.f*m;
	Tgc[8] = y;
	Tgc[9] = y + 3.f*m;
	Tgc[10] = y + 6.f*m;
	Tgc[11] = 2.f*y;
	Tgc[12] = 2.f*y + 6.f*m;
	Tgc[13] = 3.f*y;
	Tgc[14] = 3.f*y + 6.f*m;

	rgc[0] = 0.05f;
	rgc[1] = 0.07f;
	rgc[2] = 0.08f;
	rgc[3] = 0.06f;
	rgc[4] = 0.07f;
	rgc[5] = 0.1f;
	rgc[6] = 0.11f;
	rgc[7] = 0.13f;
	rgc[8] = 0.12f;
	rgc[9] = 0.14f;
	rgc[10] = 0.145f;
	rgc[11] = 0.14f;
	rgc[12] = 0.135f;
	rgc[13] = 0.13f;
	rgc[14] = 0.f*y;

	int k;
	FILE *ParFp;
	std::string TmpString;
	//Spline Volatility parameters------------------------------
	// - Read values from input file on CPU
	TmpString = "Cg.txt";
	ParFp = fopen(TmpString.c_str(), "r");
	if (ParFp == NULL) {
		fprintf(stderr, "File '%s' unreachable!\n", TmpString.c_str());
		exit(EXIT_FAILURE);
	}
	// - Store values in input data tables on CPU
	for (k = 0; k < 1120; k++) {
		if (fscanf(ParFp, "%f", &Cgc[k]) <= 0) {
			fprintf(stderr, "Error while reading file '%s'!\n", TmpString.c_str());
			exit(EXIT_FAILURE);
		}
	}
	fclose(ParFp);


	cudaMemcpyToSymbol(Kg, Kgc, nk * sizeof(float));
	cudaMemcpyToSymbol(Tg, Tgc, nt * sizeof(float));
	cudaMemcpyToSymbol(rg, rgc, nt * sizeof(float));
	cudaMemcpyToSymbol(Cg, Cgc, 16 * (nt - 1)*(nk - 1) * sizeof(float));
}

// Time index  (by prof)
__device__ int timeIdx(float t) {
	int i, I;
	for (i = 14; i >= 0; i--) {
		if (t < Tg[i]) {
			I = i;
		}
	}
	return I;
}

// Interest rate time integral (by prof)
__device__ float rt_int(float t, float T, int i, int j)
{
	float res;
	int k;
	if (i == j) {
		res = (T - t)*rg[i];
	}
	else {
		res = (T - Tg[j - 1])*rg[j] + (Tg[i] - t)*rg[i];
		for (k = i + 1; k < j; k++) {
			res += (Tg[k] - Tg[k - 1])*rg[k];
		}
	}

	return res;
}

// Monomials till third degree (by prof)
__device__ float mon(float x, int i) { return 1.0f*(i == 0) + x * (i == 1) + x * x*(i == 2) + x * x*x*(i == 3); }

// Local volatility from bicubic interpolation of implied volatility (by prof)
__device__ void vol_d(float x, float x0, float t, float *V, int q) {

	float u1 = 0.0f;
	float u2 = 0.0f;
	float d1, d2, d_1;
	float y = 0.0f;
	float y_1 = 0.0f, y_2 = 0.0f, y_22 = 0.0f;
	int k = 0;


	if (x >= Kg[5]) {
		k = 4;
		d2 = 1.0f / (Kg[k + 1] - Kg[k]);
		u2 = 1.0f;
	}
	else {
		if (x <= Kg[0]) {
			k = 0;
			d2 = 1.0f / (Kg[k + 1] - Kg[k]);
			u2 = 1.0f;
		}
		else {
			while (Kg[k + 1] < x) {
				k++;
			}
			d2 = 1.0f / (Kg[k + 1] - Kg[k]);
			u2 = (x - Kg[k]) / (Kg[k + 1] - Kg[k]);
		}
	}

	d1 = 1.0f / (Tg[q + 1] - Tg[q]);
	u1 = (t - Tg[q]) / (Tg[q + 1] - Tg[q]);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			y += Cg[k * 14 * 16 + q * 16 + j + i * 4] * mon(u1, i)*mon(u2, j);
			y_1 += i * Cg[k * 14 * 16 + q * 16 + i * 4 + j] * mon(u1, i - 1)*mon(u2, j)*d1;
			y_2 += j * Cg[k * 14 * 16 + q * 16 + i * 4 + j] * mon(u1, i)*mon(u2, j - 1)*d2;
			y_22 += j * (j - 1)*Cg[k * 14 * 16 + q * 16 + i * 4 + j] * mon(u1, i)*mon(u2, j - 2)*d2*d2;
		}
	}
	d_1 = (logf(x0 / x) + rt_int(0.0f, t, 0, q)) / (y*sqrtf(t)) + 0.5f*y*sqrtf(t);
	u1 = x * x*(y_22 - d_1 * sqrtf(t)*y_2*y_2 + (1.0f / y)*((1.0f / (x*sqrtf(t)))
		+ d_1 * y_2)*((1.0f / (x*sqrtf(t))) + d_1 * y_2));
	u2 = 2.0f*y_1 + y / t + 2.0f*x*rg[q] * y_2;

	*V = sqrtf(fminf(fmaxf(u2 / u1, 0.0001f), 0.5f));
}


// SER_c: we need a more intelligent way to compare les reals
// otherwise it'll not work if we do need to check the identity
__device__ bool equal(double x, double y) {
	return abs(x - y) < err_r;
}


// function to get interest rate which is in rgc defined by prof
__device__ float get_rgc() {
	// we should use Rgc table to get the real interest rate, however the way that the prof presents it is too complicated for 
	// extracting interest rate, we set it to zero for the moment
	return 0.f;
}


// function help Ys to decide which update schema to use
__device__ int which_Schema(int tStep) {
	float t = tStep * delta_T;
	if (equal(t, T[M - 2])) {
		return 1;// use eq before 10
	}
	else {
		for (int k = 3; k <= M; k++) {
			if (equal(t, T[M - k])) {
				return k;// use eq(10) k stands for the k + 1 in eq(10)
				// ATTENTION: the index of T[] starts from 1 in the definition of the prof while here it starts from 0
			}
		}
		return 0;// normal 
	}

}

__device__ float One_Zero(float S, float B, bool up) {
	// play the role of indicatrice in (10) and eq before (10)
	// up means all values >= B are kept, vice versa
	// SER_c: this is not what we want here...
	if (up) {
		return S < B ? 0.f : 1.f;
	}
	else {
		return S < B ? 1.f : 0.f;
	}
}

// update Ys by Schema0 (which means E[(S_T - K)_+|S_t])

__device__ void Update_by_Schema0() {
	int idx = threadIdx.x + (blockIdx.x - 1) * N; // SER_c: don't we need to multiply the second term by N? 
	if (blockIdx.x > P2 || blockIdx.x < P1) {
		Ys[idx] = 0;
		return; // SER_c: we stop
	}
	

	double x = log(Kgmin) + threadIdx.x * delta_X; // SER_c : was not defined before

	// SER_c : here might be a problem because of precision
	// so we need to leave a gap; for our setup delta_x is 2.5e-2 and we can 
	// hope that an error of 1e-4 is enough
	// we do not come until log(Kgmax): we stop just before
	// log(Kgmin) + (N - 1) * delta != log(Kgmax)

	if (x >= log(Kgmax) - delta_X - err_r) {
		// upper boundary
		float pmax = Kgmax - K;
		// when touches upper boundary, Cs[idx+1] is set to be Cs[idx]
		// SER_c:  we need to use Bs and Cs too

		Ys[idx] = -As[idx - 1] * Zs[idx - 1] + (2 - Bs[idx]) * Zs[idx] + (-Cs[idx]) * pmax;
	}
	else {
		if (x <= log(Kgmin) + err_r) { // SER_c: Again we add a gap
			// lower boundary
			float pmin = 0;
			Ys[idx] = -As[idx] * pmin + (2 - Bs[idx]) * Zs[idx] + (-Cs[idx + 1]) * Zs[idx + 1];
		}
		else {
			// in the middle
			Ys[idx] = -As[idx - 1] * Zs[idx - 1] + (2 - Bs[idx]) * Zs[idx] + (-Cs[idx + 1]) * Zs[idx + 1];
		}
		
	}
	return;
}

// update Ys by Schema1 (which means just with indicatrice, not the one with the sum two indicatrice
__device__ void Update_by_Schema1(bool up) {

	int idx = threadIdx.x + (blockIdx.x - 1) * N; // SER_c
	double x = log(Kgmin) + threadIdx.x * delta_X; // SER_c


	// SER_c: x - delta, error
	if (x >= log(Kgmax) - delta_X - err_r) {
		// upper boundary
		float pmax = Kgmax - K;
		// when touches upper boundary, As[idx+1] is set to be As[idx]
		// SER_c: As, Bs, Cs // also modified
		Ys[idx] = -As[idx - 1] * Zs[idx - 1]  * One_Zero(x - delta_X, B, up)\
		 + (2 - Bs[idx]) * Zs[idx] * One_Zero(x, B, up) +\
			(-Cs[idx]) * pmax * One_Zero(x + delta_X, B, up);
	}
	else {
		// SER_c: error
		if (x <= log(Kgmin) + err_r) {
			// lower boundary
			float pmin = 0;
			//SER_c: Bs, Cs
			Ys[idx] = -As[idx] * pmin * One_Zero(x - delta_X, B, up) + \
				Zs[idx]  * (2 - Bs[idx]) * One_Zero(x, B, up) +\
				(-Cs[idx + 1]) * Zs[idx + 1] * One_Zero(x + delta_X, B, up);
		}
		else {
			// in the middle
			//SER_c: Bs, Cs
			Ys[idx] = -As[idx - 1] * Zs[idx - 1] * One_Zero(x - delta_X, B, up)+\
				(2 - Bs[idx]) * Zs[idx]  * One_Zero(x, B, up) + \
				(-Cs[idx + 1]) * Zs[idx + 1] * One_Zero(x + delta_X, B, up);
		}
		
	}
	return;
}

// update Ys by Schema3 (which means the one in eq(10) with the sum of two indicatrice
__device__ void Update_by_Schema2(int cut) {
	int idx_this_j = threadIdx.x + (blockIdx.x - 1) * N;
	int idx_next_j = threadIdx.x + (blockIdx.x) * N;

	// SER_c: don't we need to multiply the second term by N? 

	if (blockIdx.x > P2 || blockIdx.x < cut) {
		Ys[idx_this_j] = 0;
		return; // SER_c: we stop
	}

	double x = log(Kgmin) + threadIdx.x * delta_X;

	if (x >= log(Kgmax) - delta_X - err_r) {
		// upper boundary
		float pmax = Kgmax - K;
		// when touches upper boundary, As[idx+1] is set to be As[idx]
		// SER_c: As, Bs, Cs // also modified

		// SER_c: I-st part Id(S_T >= B)
		Ys[idx_this_j] = -As[idx_this_j - 1] * Zs[idx_this_j - 1] * One_Zero(x - delta_X, B, true) + \
			(2 - Bs[idx_this_j]) * Zs[idx_this_j] * One_Zero(x, B, true) + \
			(-Cs[idx_this_j + 1]) * pmax * One_Zero(x + delta_X, B, true) + \
			// SER_c: II-nd part Id(S_T < B)
			- As[idx_next_j - 1] * Zs[idx_next_j - 1] * One_Zero(x - delta_X, B, false) + \
			(2 - Bs[idx_next_j]) * Zs[idx_next_j] * One_Zero(x, B, false) + \
			(-Cs[idx_next_j + 1]) * pmax * One_Zero(x + delta_X, B, false);
	}

	else {
		if (x <= log(Kgmin) + err_r) {
			// lower boundary
			float pmin = 0;
			// SER_c: I-st part Id(S_T >= B)
			Ys[idx_this_j] = -As[idx_this_j - 1] * pmin * One_Zero(x - delta_X, B, true) + \
				(2 - Bs[idx_this_j]) * Zs[idx_this_j] * One_Zero(x, B, true) + \
				(-Cs[idx_this_j + 1]) * Zs[idx_this_j + 1] * One_Zero(x + delta_X, B, true) + \
				// SER_c: II-nd part Id(S_T < B)
				- As[idx_next_j - 1] * pmin * One_Zero(x - delta_X, B, false) + \
				(2 - Bs[idx_next_j]) * Zs[idx_next_j] * One_Zero(x, B, false) + \
				(-Cs[idx_next_j + 1]) * Zs[idx_next_j + 1] * One_Zero(x + delta_X, B, false);
		}

		else {



			// SER_c: General case: in the middle

			// SER_c: I-st part Id(S_T >= B)
			Ys[idx_this_j] = -As[idx_this_j - 1] * Zs[idx_this_j - 1] * One_Zero(x - delta_X, B, true) + \
				(2 - Bs[idx_this_j]) * Zs[idx_this_j] * One_Zero(x, B, true) + \
				(-Cs[idx_this_j + 1]) * Zs[idx_this_j + 1] * One_Zero(x + delta_X, B, true) + \
				// SER_c: II-nd part Id(S_T < B)
				- As[idx_next_j - 1] * Zs[idx_next_j - 1] * One_Zero(x - delta_X, B, false) + \
				(2 - Bs[idx_next_j]) * Zs[idx_next_j] * One_Zero(x, B, false) + \
				(-Cs[idx_next_j + 1]) * Zs[idx_next_j + 1] * One_Zero(x + delta_X, B, false);

		}


	}

	return;
}

__device__ void Update_Ys(int tStep, int NN) {
	// Ys is the right part of the recursive equation
	// please notice that: we update first Ys for step i-1 using As, Bs, Cs, at step i, then we update As, Bs, Cs for the step i-1
	// by doing that, we don't need to create three extra array to store pd,pm,pd since they can be calculated from As,Bs,Cs as following
	// *********************************************************************************
	// As[ threadIdx.x + N*blockIdx.x ] = qd( threadIdx.x*delta_t, threadIdx.x*delta_x)
	// Bs[ threadIdx.x + N*blockIdx.x ] = qm( threadIdx.x*delta_t, threadIdx.x*delta_x)
	// Cs[ threadIdx.x + N*blockIdx.x ] = qu( threadIdx.x*delta_t, threadIdx.x*delta_x)
	// pd = - qd 
	// pm = 2 - qm
	// pu = - qu
	// *********************************************************************************
	int schema = which_Schema(tStep);

	if (schema == 0) {
		Update_by_Schema0();
	}
	if (schema == 1) {
		if (blockIdx.x == P2 || blockIdx.x == (P1 - 1)) {
			bool up = true;
			if (blockIdx.x == (P1 - 1)) {
				up = false;
			}
			Update_by_Schema1(up);
		}
		else {
			Update_by_Schema0(); // SER_c: it was 1 but we use the standard dynamic
		}
	}
	if (schema >= 2) {
		int k = schema; // +1;  SER_c: k in which_Schema already stands for "k+1" and is never equal to 2 ;
		int Pk = Max(P1 - k + 1, 0);// SER_c: k corresponds again to k + 1 in (10)
		if (blockIdx.x == P2 || blockIdx.x == (Pk - 1)) {
			bool up = true;
			if (blockIdx.x == (Pk - 1)) {
				up = false;
			}
			Update_by_Schema1(up);
		}
		else {
			int Pk = Max(P1 - k + 1, 0);
			Update_by_Schema2(Pk - 1);
		}
	}
}

__device__ void Update_As(int tStep, int NN) {
	int idx = threadIdx.x + (blockIdx.x - 1);
	// only when threadIdx.x < dimension of As do update, other threads do nothing
	if (threadIdx.x < NN - 1) {
		// set vol constant for test, since I'm not sure how to use functiong vol_d,
		// we could decomment de vol_d line to get local volatility
		float sigma = 0.2;
		float x = Kgmin + delta_X * (threadIdx.x + 1);
		// vol_d(x,x0,t,&sigma,q);
		float miu = get_rgc() - sigma;
		As[idx] = -(sigma * sigma * delta_T) / (4 * delta_X * delta_X) - miu * delta_T / (4 * delta_X);
	}
}

__device__ void Update_Bs(int tStep, int NN) {
	int idx = threadIdx.x + (blockIdx.x - 1);
	// only when threadIdx.x < dimension of Bs do update, other threads do nothing
	if (threadIdx.x < NN) { // SER_c: collision N and NN
		float sigma = 0.2;
		float x = Kgmin + delta_X * threadIdx.x;
		// vol_d(x,x0,t,&sigma,q);
		Bs[idx] = 1 + (sigma * sigma * delta_T) / (2 * delta_X * delta_X);
	}
}

__device__ void Update_Cs(int tStep, int NN) {
	int idx = threadIdx.x + (blockIdx.x - 1);
	// only when threadIdx.x < dimension of Cs do update, other threads do nothing
	if (threadIdx.x < NN - 1) { // SER_c: collision N and NN
		float sigma = 0.2;
		float x = Kgmin + delta_X * threadIdx.x;
		// vol_d(x,x0,t,&sigma,q);
		float miu = get_rgc() - sigma;
		Cs[idx] = -(sigma * sigma * delta_T) / (4 * delta_X * delta_X) + miu * delta_T / (4 * delta_X);
	}
}

__device__ void Thomas_Solver(int tStep) { // SER_c: do we really need it here?, int N
	// Thomas algorithm to solve the triangular system
	// N : size of the matrix
	// As,Bs,Cs,Ys,Zs are in the global memory of GPU which can be accessed by all threads in all block
	// blockIdx.x corresponds to the index j in eq(10)

	// since block zero works only on verifying synchronization, the blockIdx.x in Thomas_Solver starts
	// from 1, that's why we have (blockIdx.x - 1) here
	if (threadIdx.x == 0) {
		int idx = threadIdx.x + N * (blockIdx.x - 1);
		Cs[idx] = Cs[idx] / Bs[idx];
		Ys[idx] = Ys[idx] / Bs[idx];

		//only the thread 0 of each block solves the system by Thomas, since Thomas is sequential algorithm
		for (int i = 1; i < N - 1; i++) {
			Cs[idx + i] = Cs[idx + i] / (Bs[idx + i] - As[idx + i - 1] * Cs[idx + i - 1]);
		}

		for (int i = 1; i < N; i++) { // SER_c: was an error M -> N
			Ys[idx + i] = (Ys[idx + i] - As[idx + i - 1] * Ys[idx + i - 1]) / (Bs[idx + i] - As[idx + i - 1] * Cs[idx + i - 1]);
		}

		Zs[idx + N - 1] = Ys[idx + N - 1];

		int i;

		for (int i = N - 2; i > -1; i--) {
			Zs[idx + i] = Ys[idx + i] - Cs[idx + i] * Zs[idx + i + 1];
		}

	}
}
		

// main kernel solves the whole problem
__global__ void PDE_Solver() { // SER_c: de we really need int N here?
	// block zero is responsible for synchronization for other blocks
	if (blockIdx.x == 0) {
		// only the first thread in block zero is used, other threads do nothing
		if (threadIdx.x == 0) {
			for (int i = L - 1; i >= 0; i--) {
				while (synchro_count[i] != M) {
					// wait for all other blocks finishing calculation for time step t_i 
				}
				// once all other blocks have finished calculation, change the i-1'th synchronization state
				// allows all other blocks continuing the t_i-1 time step work
				is_synchronized[i - 1] = true;
			}
		}
	}
	else {
		// other blocks are responsible for calculation
		for (int i = L - 1; i >= 0; i--) {
			while (is_synchronized[i] != true) {
				// wait for the time step i's state to be changed to true
			}

			Update_Ys(i, N); // SER_c: 
			__syncthreads();// ATTENTION: all threads in the same block locked by __syncthreads until all threads arrive here

			Update_As(i, N);
			__syncthreads();

			Update_Bs(i, N);
			__syncthreads();

			Update_Cs(i, N);
			__syncthreads();

			Thomas_Solver(i);
			__syncthreads();

			// only the first thread has the right to write in synchro_count
			// since all threads in the block ar synchronized, only the first thread writes
			// can still garantie that all threads in the block have finished the task.
			if (threadIdx.x == 0) {
				atomicAdd(&synchro_count[i], 1);
			}
		}
	}
}




		int main()
		{
			
			std::cout << "Don't give up!";
			return 0;
		}


