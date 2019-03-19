#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include "swap_utils.h"
#include "linalg.h"
#include <math.h>


void init_Rho(alglib::real_2d_array& rho, double beta, double* T, int N){
    // get correlation matrix rho
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            rho[i][j] = exp( - beta * abs( T[i] - T[j] ));
        }
    }
    return;
}

void print_Matrix(alglib::real_2d_array Mat, int M, int N){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            std::cout<< Mat[i][j] << ' ';
        }
        std::cout<<std::endl;
    }
    return;
}

void get_U(alglib::real_2d_array& U, double beta, double* T, int N){
    // cholesky decomposition in alglib
    alglib::real_2d_array rho_;
    rho_.setlength(N,N);
    init_Rho(rho_, beta, T, N);
    alglib::spdmatrixcholesky(rho_, N, true);
    for(int i=0; i < N; i++){
        for(int j=0; j < N; j++){
            if(j>=i){
                U[i][j]=rho_[i][j];
            }
            else{
                U[i][j]=0;
            }
        }
    }
    return;
}

double R_Swap(alglib::real_1d_array& log_L, double delta, int N){
    //get swap rate from log Libor rates
    alglib::real_1d_array L_;
    L_.setlength(N);
    for(int i = 0; i < N; i++){
        L_[i] = exp(log_L[i]);
    }

    double num = 1.0;
    for(int j = 0; j < N; j++){
        num *= (1 + delta * L_[j]);
    }
    num = 1.0 - 1.0 / num;
    double den = 0.0;
    for(int i = 0; i < N; i++){
        double temp = 1.0;
        for(int j = 0; j <= i; j++){
            temp *= (1 + delta * L_[j]);
        }
        den += 1/temp;
    }
    den *= delta;
    return (num/den);
}

double get_Bern(double p){
    return distribution(generator) < p ? -1.0 : 1.0;
}

double get_Norm(alglib::real_1d_array& proj_logL, alglib::real_1d_array& log_L, int N, int q){
    double norm = 0;
    for(int i = 0; i < N; i++){
        if( q == 1) norm = norm + abs(proj_logL[i] - log_L[i]);
        else norm = norm + pow((proj_logL[i] - log_L[i]),q);
    }
    if(q == 1) return norm;
    else return pow(norm,1.0/q);
}

double get_p(double norm, int N, double h,double sigma){
    double lam_h = sqrt(N) * (sigma * sigma * h * N - 0.5 * sigma * sigma * h + sigma * sqrt(h * N));
    return (lam_h / (norm + lam_h));
}

void jump_Back(alglib::real_1d_array& log_L, alglib::real_1d_array& proj_logL, int N, double h, double sigma){
    double lam_h = sqrt(N) * (sigma * sigma * h * N - 0.5 * sigma * sigma * h + sigma * sqrt(h * N));
    double norm = get_Norm(proj_logL, log_L, N);
    for(int i = 0; i < N; i++){
        log_L[i] = log_L[i] + lam_h * (log_L[i] - proj_logL[i]) / norm;
    }
    return;
}

void get_RandomStep(alglib::real_1d_array& randomStep, double p, int N){
    for(int i = 0; i < N; i++){
        randomStep[i] = get_Bern(p);
    }
    return;
}

void one_Simulation(alglib::real_1d_array& log_L, alglib::real_2d_array& U, alglib::real_2d_array& Rho, double delta, double sigma, double h, int N){
    // one step update of the simulation of log_L
    alglib::real_1d_array randomStep_,L_;
    randomStep_.setlength(N);
    get_RandomStep(randomStep_, 0.5, N);
    L_.setlength(N);
    for(int i = 0; i < N; i++){
        L_[i] = exp(log_L[i]);
    }

    for(int i = 0; i < N; i++){

        double term1 = 0;
        for(int j = 0; j <= i; j++){
            term1 += ((delta * L_[j] / (1 + delta * L_[j])) * Rho[i][j] * sigma * h);
        }
        term1 *= sigma;

        double term2 = -0.5 * (sigma * sigma) * h;

        double term3 = 0;
        for(int j = i; j < N; j++){
            term3 += (U[i][j] * randomStep_[j]);
        }
        term3 *= (sigma * sqrt(h));
        
        log_L[i] = log_L[i] + term1 + term2 + term3;
    }
    return;
}

double get_MaxlogL(alglib::real_1d_array& log_L, int N){
    double max_ = log_L[0];
    for(int i = 1; i < N; i++){
        if (log_L[i] > max_) max_ = log_L[i];
    }
    return max_ ;
}

bool first_Test(alglib::real_1d_array& log_L, int N, double sigma, double h, double R_up){
    // test defined by (43)
    double log_L_max = get_MaxlogL(log_L, N);
    double log_L_hat = log_L_max + (sigma*sigma) * h * N - 0.5 * (sigma*sigma) * h + sigma * sqrt(h * N);
    if (log_L_hat < log(R_up)) return true;
    else return false;
}

bool second_Test(alglib::real_1d_array& log_L, int N, double sigma, double h, double delta, double R_up){
    // test defined by (44)
    alglib::real_1d_array log_L_;
    log_L_.setlength(N);
    for(int i = 0; i < N; i++){
        log_L_[i] = log(exp(log_L[i]) * (1 + (i+1) * sigma * sigma * h + sigma * sqrt((N-i) * h)));
    }
    double r_swap = R_Swap(log_L_, delta, N);
    if (r_swap < R_up) return true;
    else return false;
}

double get_log_L0(const alglib::real_1d_array& log_L, int N, double delta, double R_up){
    // std::cout<<"in get_log_L0" <<std::endl;
    // get log_L0 by the constraint equation
    alglib::real_1d_array L_;
    L_.setlength(N);
    for(int i = 0; i < N; i++){
        L_[i] = exp(log_L[i]);
    }
    // print_logL(L_,N);
    double den = 1.0;
    for(int j = 1; j < N; j++){
        den *= (1.0 + delta * L_[j]);
    }
    double num = 0.0;
    for(int i = 0; i < (N-1); i++){
        double temp = 1.0;
        for(int j = (i+1); j < N; j++){
            temp *= (1.0 + delta * L_[j]);
        }
        num += temp;
    }
    num = R_up * (1.0 + num) + 1.0;
    // std::cout<<" num : "<< num <<std::endl;
    // std::cout<<" den : "<< den <<std::endl;
    double log_L0 = log(num/den - 1.0/delta);
    // std::cout<<" log_L0 : "<< log_L0 <<std::endl; 
    return log_L0;
}

void fonction_fvec(const alglib::real_1d_array& x, alglib::real_1d_array& fi, void *ptr){
    struct args *arg = (struct args *)ptr;
    alglib::real_1d_array log_L = arg->log_L;
    int N = arg->N;
    double delta = arg->delta;
    double R_up = arg->R_up;
    double x_0 = arg->get(x,N,delta,R_up);
    // std::cout<< " x_0 : "<< x_0 << std::endl; 
    for(int i = 0; i < N; i++){
        if( i == 0) fi[i] = log_L[i] - x_0;
        else fi[i] =  log_L[i] - x[i];
    }
}

alglib::real_1d_array projection(alglib::real_1d_array& proj_x, void (*fvec)(const alglib::real_1d_array &x, alglib::real_1d_array &fi, void *ptr), void *ptr){
    int N = proj_x.length();

    //calculate projection using alglib::minlmoptimize
    alglib::real_1d_array scale;
    scale.setlength(N);
    for(int i = 0; i < N; i++){
        scale[i] = 1;
    }
    double epsx = 0.000000001;
    alglib::ae_int_t maxits = 0;
    alglib::minlmstate state;
    alglib::minlmreport rep;
    
    alglib::minlmcreatev(N, proj_x, 0.0001, state);
    alglib::minlmsetcond(state, epsx, maxits);
    alglib::minlmsetscale(state, scale);
    // optimization
    alglib::minlmoptimize(state,fvec,NULL,ptr);

    alglib::minlmresults(state, proj_x, rep);

    // calculate proj_x[0] with constrant equation
    struct args *arg = (struct args *)ptr;
    alglib::real_1d_array log_L = arg->log_L;
    double delta = arg->delta;
    double R_up = arg->R_up;
    proj_x[0] = arg->get(proj_x,N,delta,R_up);
    // print_logL(proj_x,N);
    return proj_x;
}

void simulation(alglib::real_1d_array& log_L, alglib::real_2d_array& U, alglib::real_2d_array& Rho, double delta, double sigma, double h, int N, double T0,int itr, bool is_Recording){
    std::ofstream record("record_trajectoire.txt");
    for(int i = 0; i < itr; i++){
        double timeStamp = i * T0 / itr;  
        if(is_Recording) record_logL(log_L, N, timeStamp, record);
        one_Simulation(log_L, U, Rho, delta, sigma, h, N);
    }
    record.close();
    return;
}

outCome real_simulation(alglib::real_1d_array& log_L, alglib::real_2d_array& U, alglib::real_2d_array& Rho, double R_up, double delta, double sigma, double h, int N, double T0, int itr, std::ofstream& record, bool is_Recording){
    double timeStamp = 0;
    for(int i = 0; i < itr; i++){
        timeStamp = (i+1) * T0 / itr;
        if(first_Test(log_L, N, sigma, h, R_up)){
            one_Simulation(log_L, U, Rho, delta, sigma, h, N);
        }else{
            if(second_Test(log_L, N, sigma, h, delta, R_up)){
                one_Simulation(log_L, U, Rho, delta, sigma, h, N);
            }else{
                alglib::real_1d_array proj_x;
                proj_x.setlength(N);
                for(int i = 0; i < N; i++){
                    proj_x[i] = -3.0;
                }
                struct args arg = {N, R_up, delta, log_L ,get_log_L0};
                alglib::real_1d_array proj_logL = projection(proj_x, fonction_fvec, &arg);
                double norm = get_Norm(proj_logL, log_L, N);
                double p = get_p(norm, N, h, sigma);
                if(get_Bern(p)<0){
                    break;
                }
                else{
                    jump_Back(log_L, proj_logL, N, h, sigma);
                    one_Simulation(log_L, U, Rho, delta, sigma, h, N);
                }
            }
        }
    }
    if(is_Recording) record_logL(log_L, N, timeStamp, record);
    outCome result = {timeStamp, log_L};
    return result;
}

alglib::real_1d_array get_zeroCoupon(alglib::real_1d_array& log_L, int N, double delta){
    alglib::real_1d_array zeroCoupon;
    zeroCoupon.setlength(N);
    for(int i = 0; i < N; i++){
        if(i == 0){
            zeroCoupon[i] = 1.0 / (delta * exp(log_L[i]) + 1);
        }else{
            zeroCoupon[i] = zeroCoupon[i-1] / (delta * exp(log_L[i]) + 1);
        }
    }
    return zeroCoupon;
}

double get_Price(alglib::real_1d_array& zeroCoupon, double Rswap, double K, double delta, int N){
    double price = delta * (Rswap - K);
    if(price <= 0){
        return price;
    }
    double coupon = 0;
    for(int i = 0; i < N; i++){
        coupon += zeroCoupon[i];
    }
    return (price * coupon);
}

void monte_carlo(int num_simulation, alglib::real_1d_array& log_L, alglib::real_2d_array& U, alglib::real_2d_array& Rho, double R_up, double delta, double sigma, double h, int N, double T0, double actualisation, double K, std::ofstream& record, bool is_Recording){
    int itr = (int) T0 / h;
    double exit_Time = 0;
    double price = 0;
    int n = 0;
    for(int i = 0; i < num_simulation; i++){
        //each simulation set the begining point at log_L
        alglib::real_1d_array log_L_;
        log_L_.setlength(N);
        for(int j = 0; j < N; j++){
            log_L_[j] = log_L[j];
        }
        outCome result = real_simulation(log_L_, U, Rho, R_up, delta, sigma, h, N, T0, itr, record, false);
        double time = result.timeStamp;
        if (time < T0){
            exit_Time += time;
        }else{
            exit_Time += time;
            double Rswap = R_Swap(result.logL, delta, N);
            if(isnan(Rswap)){
                n++;
                continue;
            }
            alglib::real_1d_array zeroCoupon = get_zeroCoupon(result.logL, N, delta);
            price = price + actualisation * get_Price(zeroCoupon, Rswap, K, delta, N);
        }
    }
    double mean_exitTime = exit_Time / num_simulation;
    double mean_price = price / num_simulation;
    std::cout <<"mean exit time : " << mean_exitTime << std::endl;
    std::cout <<"price : " << mean_price << std::endl;
    std::cout <<"invalid simulation number : "<< n << std::endl;
}

void print_logL(alglib::real_1d_array& log_L, int N){
    std::cout<<"[";
    for(int i = 0; i < N; i++){
        if(i != 0) std::cout<<" ";
        std::cout<< log_L[i];
        if(i != (N-1)) std::cout<<",";
    }
    std::cout<<"]"<<std::endl;
    return;
}

void record_logL(alglib::real_1d_array& log_L, int N, double timeStamp,std::ofstream& record){
    record<<timeStamp<<" ";
    for(int i = 0; i < N; i++){
        if (i != (N-1)) record<<log_L[i]<<" ";
        else record<<log_L[i];
    }
    record<<std::endl;
}


