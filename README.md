# MCProject

The project contains two parts. Each part should be compiled and executed separetely. 

# Barrier Cap/Floor

To compile: g++ -I. -o cap.out *.cpp

To launch: ./cap.out

Files:

main.cpp: main function for Barrier Cap

utils.cpp: contains all the necessary stuff and implementation of function

utils.h: header file

The code prints the results directly into console.

The complete time of execution may take upto 30 minutes.
If one needs to reduce the execution time, please change values of the variables num_sim and N in main.cpp (e.g. num_sim = 10000, N = 100)

# Barrier Swaption
### Calculate Barrier Swaption by Monte-Carlo method
Don't need to do anything, just go into the directory of Swaption then type the compilation command
Compilation command : 
g++ -I. -o swap.out *.cpp

You will find three files written by us:

swap_utils.h : header file 

swap_utils.cpp : all functions need for the Barrier Swaption

test_swap.cpp : main function for the Barrier Swaption

All other files are form alglib, they are used for the optimization procedure in the algorithm.

The code prints out the prices and runing times for each discretization step at the terminal screen.
As output, you are going the expect a file which stores the exit time and the result of each simulation per discretization step.
