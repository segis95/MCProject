#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include "swap_utils.h"
#include "ap.h"
#include "linalg.h"

// Nothing should be changed

std::random_device generator;
std::uniform_real_distribution<double> distribution(0.0, 1.0);

int main(void){
    // ********************************************//
    // initialization all variables
    // ********************************************//
    alglib::ae_int_t N = 10;
    double T_ = 10;
    double beta = 0.1;
    double delta = 1.0;
    double sigma = 0.1;
    double h = 0.1;
    double R_up = 0.075;
    double K = 0.01;
    double* T = new double[N];
    // initial tenor date
    for(int i = 0; i < N; i++){
        T[i] = T_ + i;
    }

    // ********************************************//
    // creat matrix rho
    // ********************************************//
    alglib::real_2d_array Rho;
    Rho.setlength(N, N);
    init_Rho(Rho, beta, T, N);
    // print_Matrix(rho,N,N);

    // ********************************************//
    // creat matrix U , do cholesky to rho
    // ********************************************//
    alglib::real_2d_array U;
    U.setlength(N, N);
    get_U(U, beta, T, N);
    // print_Matrix(U,N,N);

    // ********************************************//
    // test Libor rate
    // ********************************************//
    // alglib::real_1d_array log_L("[-2.99573227, -2.99573227, -2.99573227, -2.99573227, -2.99573227]");
    // double r_Swap = R_Swap(log_L, delta, 5);
    // std::cout << r_Swap << std::endl;

    // ********************************************//
    // test one step simulation
    // ********************************************//
    // print_logL(log_L,5);
    // one_Simulation(log_L,U,Rho,delta,sigma,h,5);
    // print_logL(log_L,5);
    // int itr = 1000;
    // simulation(log_L, U, Rho, delta, sigma, h, N, T_, itr, true);
    
    
    // ********************************************//
    // test projection
    // input :
    // [-2.73936103, -2.85525712, -2.9555008 , -3.01048378, -2.92122347]
    // expected output : 
    // [-2.33568611, -2.5451155 , -2.70690741, -2.79784462, -2.70533224]
    // output :
    // [-2.33568, -2.54506, -2.70691, -2.79789, -2.70539]
    // ********************************************//
    // alglib::real_1d_array log_L1("[-2.73936103, -2.85525712, -2.9555008 , -3.01048378, -2.92122347]");
    // struct args arg = {N, R_up, delta, log_L1 ,get_log_L0};
    // alglib::real_1d_array proj_L;
    // proj_L.setlength(N);
    // for(int i = 0; i < N; i++){
        // proj_L[i] = -3.0;
    // }
    // proj_L = projection(proj_L, fonction_fvec, &arg);
    // print_logL(proj_L,N);


    // ********************************************//
    // Test monte-carlo
    // ********************************************//
    alglib::real_1d_array log_L;
    log_L.setlength(N);
    for(int i = 0; i < N; i++){
        log_L[i] = log(0.05);
    }
    double actualisation = 1 / pow(1.05,10);
    double h_[6] = {0.25,0.2,0.125,0.1,0.0625,0.03125};
    std::string name[6] = {"test25_4.txt","test2_4.txt","test125_4.txt","test1_4.txt","test625_4.txt","test3125_4.txt"};
    std::string name1[6] = {"test25_5.txt","test2_5.txt","test125_5.txt","test1_5.txt","test625_5.txt","test3125_5.txt"};
    for(int i = 0; i < 6; i++){
        int num_traj = 1000000;
        std::ofstream record(name[i]);
        std::cout << "*******************************************************" << std::endl;
        std::cout << "Simulation : " << num_traj << " , Step size : " << h_[i] << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        monte_carlo_func(num_traj, log_L, U, Rho, R_up, delta, sigma, h_[i], N, T[0], actualisation, K,record,real_simulation_simple_MC,true,true);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "running time : " << duration.count() << " sec" << std::endl;
        record.close();
        std::ofstream record1(name1[i]);
        std::cout << "*******************************************************" << std::endl;
        std::cout << "Simulation 1 : " << num_traj << " , Step size : " << h_[i] << std::endl;
        auto start1 = std::chrono::high_resolution_clock::now();
        monte_carlo_func(num_traj, log_L, U, Rho, R_up, delta, sigma, h_[i], N, T[0], actualisation, K,record1,real_simulation,true,true);
        auto end1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::seconds>(end1 - start1);
        std::cout << "running time : " << duration1.count() << " sec" << std::endl;
        record1.close();
    }
    
}
