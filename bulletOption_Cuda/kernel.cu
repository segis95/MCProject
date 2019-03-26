
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "rng.h"

#define nt 15
#define nk 6
#define L 100 //time step number
#define M 9 // payment date number
#define N 100// space step number


// read only variables in GPU defined by prof, used by local volatility
__constant__ float Tg[nt];
__constant__ float rg[nt];// interest rate, constant par morceaux
__constant__ float Kg[nk];
__constant__ float Cg[16*(nt-1)*(nk-1)];


__constant__ float Kgmax = 250.f;
__constant__ float Kgmin = 20.f;
__constant__ float K = 100.f;
__constant__ int P1 = 2;
__constant__ int P2 = 8;
__constant__ float B = 120.f;
// we should define the maturity here 
// since the interest rate defined in rg has max time about 3 years, so we define maturity as 3 years
// so that in our simulation, all time steps has a defined interest rate
__constant__ float T = 3.f;
__constant__ float delta_T = T / L;
__constant__ float delta_X = (log(Kgmax)-log(Kgmin)) / N;// ATTENTION: X is log
// prefixed dates correspond to maturity T = 3
__constant__ float T[M] = {0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7};

__device__ bool is_synchronized[L];
__device__ int synchro_count[L];
__device__ float As[M*(N-1)];
__device__ float Bs[M*N];
__device__ float Cs[M*(N-1)];
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
	Cgc = (float *)calloc(16*(nk-1)*(nt-1), sizeof(float));

	// variables used for solve_PDE kernel
	is_synchronized_c = (bool *)calloc(L, sizeof(bool));
	synchro_count_c = (int *)calloc(L, sizeof(int));
	Asc = (float *)calloc(M*(N-1), sizeof(float));
	Bsc = (float *)calloc(M*N, sizeof(float));
	Csc = (float *)calloc(M*(N-1, sizeof(float));
	Ysc = (float *)calloc(M*N,sizeof(float));
	Zsc = (float *)calloc(M*N,sizeof(float));

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


float Max(float X, float Y){
	return X < Y ? X : Y;
}
// Initialize all parameters
void parameters()
{
	// initialization all parameters for PDE_solver
	for(int i = 0; i < L; i++){
		is_synchronized_c = false;
	}
	is_synchronized_c[L-1] = true;
	for(int i = 0; i < L; i++){
		synchro_count_c[i] = 0;
	}
	for(int i = 0; i < M*N; i++){
		Bsc[i] = 0.f;
		Ysc[i] = 0.f;
	}
	// initialization all Z to terminal condition
	for(int i = 0; i < N; i++){
		for(int j= 0; j < M; j++){
			if(j < p1 || j > p2){
				Zsc[i + j * N] = 0.f;
			}else{
				double x = log(Kgmin) + i * delta_X;
				Zsc[i + j * N] = Max(exp(x) - K,0.f);
			}
		}
	}
	for(int i = 0; i < M*(N-1); i++){
		Asc[i] = 0.f;
		Csc[i] = 0.f;
	}

	cudaMemcpyToSymbol(is_synchronized, is_synchronized_c, L*sizeof(bool));
	cudaMemcpyToSymbol(synchro_count, synchro_count_c, L*sizeof(int));
	cudaMemcpyToSymbol(As, Asc, M*(N-1)*sizeof(float));
	cudaMemcpyToSymbol(Cs, Csc, M*(N-1)*sizeof(float));
	cudaMemcpyToSymbol(Bs, Bsc, M*N*sizeof(float));
	cudaMemcpyToSymbol(Ys, Ysc, M*N*sizeof(float));
	cudaMemcpyToSymbol(Zs, Zsc, M*N*sizeof(float));


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
 	Tgc[10] =y + 6.f*m;
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
	string TmpString;
	//Spline Volatility parameters------------------------------
	// - Read values from input file on CPU
	TmpString = "Cg.txt";
	ParFp = fopen(TmpString.c_str(),"r");
	if (ParFp == NULL) {
	  fprintf(stderr,"File '%s' unreachable!\n",TmpString.c_str());
	  exit(EXIT_FAILURE);   
	}
	// - Store values in input data tables on CPU
	for (k = 0; k < 1120; k++) {
		if (fscanf(ParFp,"%f",&Cgc[k]) <= 0) {
		  fprintf(stderr,"Error while reading file '%s'!\n",TmpString.c_str());
		  exit(EXIT_FAILURE);          
		}
	}
	fclose(ParFp);

	
	cudaMemcpyToSymbol(Kg, Kgc, nk*sizeof(float));
	cudaMemcpyToSymbol(Tg, Tgc, nt*sizeof(float));
	cudaMemcpyToSymbol(rg, rgc, nt*sizeof(float));
	cudaMemcpyToSymbol(Cg, Cgc, 16*(nt-1)*(nk-1)*sizeof(float));
}

// Time index  (by prof)
__device__ int timeIdx(float t) {
	int i, I;
	for (i=14; i>=0; i--) {
		if(t<Tg[i]){
			I = i;
		}
	}
	return I;
}

// Interest rate time integral (by prof)
__device__ float rt_int(float t,  float T, int i, int j)
{
	float res;
	int k;
	if(i==j){
		res = (T-t)*rg[i];
	}else{
		res = (T-Tg[j-1])*rg[j] + (Tg[i]-t)*rg[i];
		for(k=i+1; k<j; k++){
			res += (Tg[k]-Tg[k-1])*rg[k];
		}
	}

	return res;
}

// Monomials till third degree (by prof)
__device__ float mon(float x, int i){return 1.0f*(i==0) + x*(i==1) + x*x*(i==2) + x*x*x*(i==3);}

// Local volatility from bicubic interpolation of implied volatility (by prof)
__device__ void vol_d(float x, float x0, float t, float *V, int q){

	float u1 = 0.0f;
	float u2 = 0.0f;
	float d1, d2, d_1;
	float y = 0.0f;
	float y_1 = 0.0f, y_2 = 0.0f, y_22 = 0.0f;
	int k = 0;
	
	
	if (x >= Kg[5]){
		k = 4;
		d2 = 1.0f /(Kg[k + 1] - Kg[k]);
		u2 = 1.0f;
	}else{
		if (x <= Kg[0]){
			k = 0;
			d2 = 1.0f/(Kg[k + 1] - Kg[k]);
			u2 = 1.0f;
		}else{
			while (Kg[k+1] < x){
				k++;
			}
			d2 = 1.0f/(Kg[k+1] - Kg[k]);
			u2 = (x - Kg[k])/(Kg[k+1] - Kg[k]);
		}
	}

	d1 = 1.0f/(Tg[q + 1] - Tg[q]);
	u1 = (t - Tg[q])/(Tg[q + 1] - Tg[q]);

	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			y += Cg[k * 14 * 16 + q * 16 + j + i * 4] * mon(u1, i)*mon(u2, j);
			y_1 += i *Cg[k * 14 * 16 + q * 16 + i * 4 + j] * mon(u1, i-1)*mon(u2, j)*d1;
			y_2 += j*Cg[k * 14 * 16 + q * 16 + i * 4 + j] * mon(u1, i)*mon(u2, j-1)*d2;
			y_22 += j *(j - 1)*Cg[k * 14 * 16 + q * 16 + i * 4 + j] * mon(u1, i)*mon(u2, j-2)*d2*d2;
		}
	}
	d_1 = (logf(x0/x) + rt_int(0.0f, t, 0, q))/(y*sqrtf(t)) + 0.5f*y*sqrtf(t);
	u1 = x*x*(y_22 - d_1*sqrtf(t)*y_2*y_2 + (1.0f/y)*((1.0f/(x*sqrtf(t))) 
		+ d_1*y_2)*((1.0f /(x*sqrtf(t))) + d_1*y_2));
	u2 = 2.0f*y_1 + y /t + 2.0f*x*rg[q]*y_2;
	
	*V = sqrtf(fminf(fmaxf(u2/u1,0.0001f),0.5f));
}


cudaError_t thomasWithCuda(double *Zs, double *As, double *Bs, double *Cs, double *Ys, int *lengths, int *starts,  int N, int size);



__global__ void thomasKernel(double *Zs, double *As, double *Bs, double *Cs, double *Ys, int *lengths, int *starts){

	int t_idx = threadIdx.x;
	int M = lengths[t_idx];
	double *a = &As[starts[t_idx] - t_idx];
	double *b = &Bs[starts[t_idx]];
	double *c = &Cs[starts[t_idx] - t_idx];
	double *y = &Ys[starts[t_idx]];
	double *z = &Zs[starts[t_idx]];

	c[0] = c[0] / b[0];
	y[0] = y[0] / b[0];


	for (int i = 1; i < M - 1; i++){
		c[i] = c[i] / (b[i] - a[i - 1] * c[i - 1]);	
	}

	for (int i = 1; i < M ; i++){
		y[i] = (y[i] - a[i - 1] * y[i - 1]) / (b[i] - a[i - 1] * c[i - 1]);
	}
	
		z[M - 1] = y[M - 1];

	for (int i = M - 2; i > -1; i--){
		z[i] = y[i] - c[i] * z[i + 1];

	}

}

// function to get interest rate which is in rgc defined by prof
__device__ float get_rgc(){
	// we should use Rgc table to get the real interest rate, however the way that the prof presents it is complicate for 
	// extracting interest rate, we set it to zero for the moment
	return 0.f;
}


// function help Ys to decide which update schema to use
__device__ int which_Schema(int tStep){
	float t = tStep * delta_T;
	if(t == T[M-2]){
		return 1;// use eq before 10
	}else{
		for(int k = 3; k <= M; k++){
			if(t == T[M-i]){
				return k;// use eq(10) k stands for the k + 1 in eq(10)
				// ATTENTION: the index of T[] starts from 1 in the definition of the prof while here it starts from 0
			}
		}
		return 0;// normal 
	}

}

__device__ float One_Zero(float X, float B, bool up){
	// play the role of indicatrice in (10) and eq before (10)
	// up means all values >= B are kept, vice versa
	if(up){
		return X < B ? 0.f : X;
	}else{
		return X < B ? X : 0.f;
	}
}

// update Ys by Schema0 (which means E[(S_T - K)_+|S_t])
__device__ void Update_by_Schema0(){
	int idx = threadIdx.x + (blockIdx.x - 1);
	if (blockIdx.x > P2 || blockIdx.x < P1){
		// if 
		Ys[idx] = 0;
	}
	if(x >= log(Kgmax)){
		// upper boundary
		float pmax = Kgmax - K;
		// when touches upper boundary, As[idx+1] is set to be As[idx]
		Ys[idx] = - As[idx-1] * Zs[idx-1] + (2 - As[idx]) * Zs[idx] + (- As[idx]) * pmax;
	}else{
		if(x <= log(Kgmin)){
			// lower boundary
			float pmin = 0
			Ys[idx] = - As[idx] * pmin + (2 - As[idx]) * Zs[idx] + (- As[idx+1]) * Zs[idx+1];
		}
		else{
			// in the middle
			Ys[idx] = - As[idx-1] * Zs[idx-1] + (2 - As[idx]) * Zs[idx] + (- As[idx+1]) * Zs[idx+1];
		}
		return;
}

// update Ys by Schema1 (which means just with indicatrice, not the one with the sum two indicatrice
__device__ void Update_by_Schema1(bool up){
	int idx = threadIdx.x + (blockIdx.x - 1);
	if(x >= log(Kgmax)){
		// upper boundary
		float pmax = Kgmax - K;
		// when touches upper boundary, As[idx+1] is set to be As[idx]
		Ys[idx] = - As[idx-1] * One_Zero(Zs[idx-1],B) + (2 - As[idx]) * One_Zero(Zs[idx],B) + (- As[idx]) * One_Zero(pmax,B);
	}else{
		if(x <= log(Kgmin)){
			// lower boundary
			float pmin = 0
			Ys[idx] = - As[idx] * One_Zero(pmin,B) + (2 - As[idx]) * One_Zero(Zs[idx],B) + (- As[idx+1] ) * One_Zero(Zs[idx+1],B);
		}
		else{
			// in the middle
			Ys[idx] = - As[idx-1] * One_Zero(Zs[idx-1],B) + (2 - As[idx]) * One_Zero(Zs[idx],B) + (- As[idx+1]) * One_Zero(Zs[idx+1],B);
		}
		return;
}

// update Ys by Schema3 (which means the one in eq(10) with the sum of two indicatrice
__device__ void Update_by_Schema2(){
	// TODO
}

__device__ void Update_Ys(int tStep, int N){
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

	if(schema == 0 ){
		Update_by_Schema0();
	}
	if(schema == 1){
		if(blockIdx.x == P2 || blockIdx.x == (P1 - 1)){
			bool up = true;
			if(blockIdx.x == (P1 -1)){
				up = false;
			}
			Update_by_Schema1(up);
		}else{
			Update_by_Schnema1();
		}
	}
	if(schema >= 2){
		int k = schema + 1;
		int Pk = Max(P1 - k, 0);
		if(blockIdx.x == P2 || blockIdx.x == (PK-1)){
			bool up = true;
			if(blockIdx.x == (Pk-1)){
				up = false;
			}
			Update_by_Schema1(up);
		}esle{
			Update_by_Shcema2();
		}
	}
}

__device__ void Update_As(int tStep, float x0, float t, int N, int q){
	int idx = threadIdx.x + (blockIdx.x - 1);
	// only when threadIdx.x < dimension of As do update, other threads do nothing
	if(threadIdx.x < N-1){
		// set vol constant for test, since I'm not sure how to use functiong vol_d,
		// we could decomment de vol_d line to get local volatility
		float sigma = 0.2;
		float x = Kgmin + delta_X * (threadIdx.x + 1);
		// vol_d(x,x0,t,&sigma,q);
		float miu = get_rgc() - sigma;
		As[idx] = - (sigma * sigma * delta_T)/(4 * delta_X * delta_X) - miu * delta_T / (4 * delta_X);
	}
}

__device__ void Update_Bs(int tStep, float x0, float t, int N, int q){
	int idx = threadIdx.x + (blockIdx.x - 1);
	// only when threadIdx.x < dimension of Bs do update, other threads do nothing
	if(threadIdx.x < N){
		float sigma = 0.2;
		float x = Kgmin + delta_X * threadIdx.x;
		// vol_d(x,x0,t,&sigma,q);
		Bs[idx] = 1 + (sigma * sigma * delta_T) / (2 * delta_X * delta_X);
	}
}

__device__ void Update_Cs(int tStep, float x0,float t,int N, int q){
	int idx = threadIdx.x + (blockIdx.x - 1);
	// only when threadIdx.x < dimension of Cs do update, other threads do nothing
	if(threadIdx.x < N - 1){
		float sigma = 0.2;
		float x = Kgmin + delta_X * threadIdx.x;
		// vol_d(x,x0,t,&sigma,q);
		float miu = get_rgc() - sigma ;
		Cs[idx] = - (sigma * sigma * delta_T)/(4 * delta_X * delta_X) + miu * delta_T / (4 * delta_X);
	}
}

__device__ void Thomas_Solver(int tStep, int N){
	// Thomas algorithm to solve the triangular system
	// N : size of the matrix
	// As,Bs,Cs,Ys,Zs are in the global momery of GPU which can be accessed by all threads in all block
	// blockIdx.x corresponds to the index j in eq(10)

	// since block zero works only on verifying synchronization, the blockIdx.x in Thomas_Solver starts
	// from 1, that's why we have (blockIdx.x - 1) here
	if(threadIdx.x == 0){
		int idx = threadIdx.x + N * (blockIdx.x - 1);
		Cs[idx] = Cs[idx] / Bs[idx];
		Ys[idx] = Ys[idx] / Bs[idx];
		//only the thread 0 of each block solve the system by Thomas, since Thomas is sequencial algorithme
		for (int i = 1; i < N - 1; i++){
			Cs[ idx+i ] = Cs[ idx+i ] / (Bs[ idx+i ] - As[ idx+i-1 ] * Cs[ idx+i-1 ]);
		}

		for (int i = 1; i < M ; i++){
			Ys[ idx+i ] = (Ys[ idx+i ] - As[ idx+i-1 ] * Ys[ idx+i-1 ]) / (Bs[ idx+i ] - As[ idx+i-1 ] * Cs[ idx+i-1 ]);
		}

		Zs[ idx+N-1 ] = Ys[ idx+N-1 ];
	
		for (int i = N - 2; i > -1; i--){
			Zs[ idx+i ] = Ys[ idx+i ] - Cs[ idx+i ] * Zs[ idx+i+1 ];
		}
	}
}

// main kernel solves the whole problem
__global__ void PDE_Solver(int N){
	// block zero is responsible for synchronization for other blocks
	if(blockIdx.x == 0){
		// only the first thread in block zero is used, other threads do nothing
		if(threadIdx.x == 0){
			for(int i = L-1; i >= 0; i--){
				while(synchro_count[i] != M){
					// wait for all other blocks finishing calculation for time step t_i 
				}
				// once all other blocks have finished calculation, change the i-1'th synchronization state
				// allows all other blocks continuing the t_i-1 time step work
				is_synchronized[i-1] = true;
			} 
		}
	}
	else{
		// other blocks are responsible for calculation
		for(int i = L-1; i >= 0; i--){
			while(is_synchronized[i] != true){
				// wait for the time step i's state to be changed to true
			}
			
			Update_Ys(i,N);
			__syncthreads();// ATTENTION: all threads in the same block locked by __syncthreads until all threads arrive here
			
			Update_As(i,N);
			__syncthreads();

			Update_Bs(i,N);
			__syncthreads();

			Update_Cs(i,N);
			__syncthreads();

			Thomas_Solver(i,N);
			__syncthreads();

			// only the first thread has the right to write in synchro_count
			// since all thread in the block is synchronized, only the first thread writes
			// can still garantie that all threads in the block have finished the task.
			if(threadIdx.x == 0){
				atomicAdd(&synchro_count[i],1);
			}
		}
	}
}


// defined by Sergey, to extract matrix information from file.txt
double* extract_udiag(double ** matrix, int M){

	double *arr = (double*)malloc((M - 1) * sizeof(double));
	for (int i = 1; i < M; i++){
		arr[i - 1] = matrix[i][i - 1]; 
	}
	return arr;
}

double* extract_diag(double ** matrix, int M){

	double *arr = (double*)malloc((M) * sizeof(double));
	for (int i = 0; i < M; i++){
		arr[i] = matrix[i][i]; 
	}
	return arr;
}

double* extract_adiag(double ** matrix, int M){

	double *arr = (double*)malloc((M - 1) * sizeof(double));
	for (int i = 1; i < M; i++){
		arr[i - 1] = matrix[i - 1][i]; 
	}
	return arr;
}



int main()
{
	std::ifstream inFile;
	inFile.open("C:/Users/Sergey/Documents/visual studio 2012/Projects/trial-cuda/systems.txt");
	int N;
	int M;

	inFile >> N; // number of systems
	double ***Systems = (double***)malloc(N * sizeof(double**));
	int *lengths = (int*)malloc(N * sizeof(int));
	double **Solutions = (double**)malloc(N * sizeof(double*)); 


	for(int i = 0; i < N; i++){
		inFile >> M;

		lengths[i] = M;
		Systems[i] = (double**)malloc(M * sizeof(double *));

		for (int j = 0; j < M; j++){
			Systems[i][j] = (double*)malloc(M * sizeof(double));
			for (int k = 0; k < M; k++){
				inFile >> Systems[i][j][k];
			}
		}

		Solutions[i] = (double*)malloc(M * sizeof(double));
		for (int j = 0; j < M; j++){
			inFile >> Solutions[i][j];
		}
	}

	inFile.close();
	int *starts = (int*)malloc(N * sizeof(int));

	int count = 0;
	for (int i = 0; i < N; i++){
		starts[i] = count;
		count += lengths[i];
		
	}

	//EXTRACTING DIAGONALS
	double *As = (double*)malloc((count - N) * sizeof(double));
	double *Bs = (double*)malloc(count * sizeof(double));
	double *Cs = (double*)malloc((count - N) * sizeof(double));
	double *Ys = (double*)malloc(count * sizeof(double));
	double *Zs = (double*)malloc(count * sizeof(double));

	double *extract1, *extract2, *extract3;
	for (int i = 0; i < N; i++){
		extract1 = extract_udiag(Systems[i], lengths[i]);
		extract2 = extract_diag(Systems[i], lengths[i]);
		extract3 = extract_adiag(Systems[i], lengths[i]);

		for (int j = starts[i]; j < starts[i] + lengths[i]; j++){
			Bs[j] = extract2[j - starts[i]];
			Ys[j] = Solutions[i][j - starts[i]];
			
		}

		for (int j = starts[i] - i; j < starts[i] + lengths[i] - (i + 1); j++){
			As[j] = extract1[j - (starts[i] - i)];
			Cs[j] = extract3[j - (starts[i] - i)];

		}

	}
	
	cudaError_t cudaStatus = thomasWithCuda(Zs, As, Bs, Cs, Ys, lengths, starts, N, count);
	cudaStatus = cudaDeviceReset();

	for (int i = 0; i < N; i++){
		for (int j = starts[i]; j < starts[i] + lengths[i]; j++)
			std::cout << Zs[j] << ' ';
		std::cout << '\n';
	}
		


	free(As);
	free(Bs);
	free(Cs);
	free(Zs);
	free(Ys);
	free(starts);
	free(Systems);
	free(lengths);
	free(Solutions);



    return 0;
}

cudaError_t thomasWithCuda(double *Zs, double *As, double *Bs, double *Cs, double *Ys, int *lengths, int *starts,  int N, int size)
{
	double *As_c;
	double *Bs_c;
	double *Cs_c;
	double *Zs_c;
	double *Ys_c;
	int *lengths_c;
	int *starts_c;


	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(1);

	cudaStatus = cudaMalloc((void**)&As_c, (size - N) * sizeof(double));
	cudaStatus = cudaMalloc((void**)&Bs_c, size * sizeof(double));
	cudaStatus = cudaMalloc((void**)&Cs_c, (size - N) * sizeof(double));
	cudaStatus = cudaMalloc((void**)&Zs_c, size * sizeof(double));
	cudaStatus = cudaMalloc((void**)&Ys_c, size * sizeof(double));
	cudaStatus = cudaMalloc((void**)&lengths_c, N * sizeof(int));
	cudaStatus = cudaMalloc((void**)&starts_c, N * sizeof(int));


	cudaStatus = cudaMemcpy(As_c, As, (size - N) * sizeof(double), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(Bs_c, Bs, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(Cs_c, Cs, (size - N) * sizeof(double), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(Ys_c, Ys, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(lengths_c, lengths, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(starts_c, starts, N * sizeof(int), cudaMemcpyHostToDevice);

	thomasKernel<<<1, N>>>(Zs_c, As_c, Bs_c, Cs_c, Ys_c, lengths_c, starts_c);

	cudaStatus = cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(Zs, Zs_c, size * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(As_c);
	cudaFree(Bs_c);
	cudaFree(Cs_c);
	cudaFree(Zs_c);
	cudaFree(Ys_c);
	cudaFree(lengths_c);
	cudaFree(starts_c);

	return cudaStatus;
}

