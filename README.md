# MCProject
# Barrier Cap/Floor

# Barrier Swaption
### Calculate Barrier Swaption by Monte-Carlo method
Don't need to do anything, just go into the directory of Swaption then type the compilation command
Compilation command : 
g++ -I. -o swap.out *.cpp

You will find three files written by us:

swap_utils.h : head file 

swap_utils.cpp : all functions need for the Barrier Swaption

test_swap.cpp : main function for the Barrier Swaption

All other files are form alglib, they are used for the optimization procedure in the algorithm.

The code prints out the prices and runing times for each discretization step at the terminal screen.
As output, you are going the expect a file which stores the exit time and the result of each simulation per discretization step.
