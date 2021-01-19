# hpcvss

On the efficient computing of long term reliable trajectories
for the Lorenz system


To run our programs one needs MPIGMP library of Tomonori Kouya. This library can be freely downloaded from:   
http://na-inet.jp/na/bnc/

------------------------------------------------------------------------------
Hybrid1_as.c program is that explained in the paper. It computes x[i+1], y[i+1], z[i+1] and  x[0], y[0], z[0] independently in parallel and also overlaps MPI_ALLREDUCE with the operations for x[i+1], y[i+1], z[i+1] that can be taken in advance.


------------------------------------------------------------------------------
Hybrid2_as.c program is a small improvement of Hybrid1_as.c

Using SPMD programming pattern we make half of the threads to compute one of the sums from the algorithm and the other half to compute the second sum. We have a little performance benefit, because for the small values of the index i the unused threads will be less and also the difference from the perfect load balance between threads will be less. 

However this approach is not general, because it strongly depends
on the number of sums for reduction (two in the particular case of the Lorenz system) and the number of available threads.


