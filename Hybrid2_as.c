#include <stdio.h>
#include <math.h>
#include <gmp.h>
#include <omp.h>
#include <mpi.h>
#include "mpi_gmp.h"
#define MPF_PREC 6976
#define N 2400
#define Numt 16
#define pad 8



int main(int argc, char **argv)
{

    FILE *outfile;
    outfile = fopen("output_hybrid_b_as.txt", "w");

    int rank, size, provided;
    MPI_Status status;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED,  &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char ch[12];
    int i,j,tid,n,l,shift,sh,istart,ifinal,nt,ifinalt,istartt;
    double elapsed_time;
    long int Exp;
    double d,t1,t2;

    mpf_set_default_prec(MPF_PREC);
    commit_mpf(&(MPI_MPF), MPF_PREC, MPI_COMM_WORLD);
    create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, MPI_COMM_WORLD);

    void *packed_mpf_s, *packed_mpf_r;
    packed_mpf_s  = allocbuf_mpf(MPF_PREC, 2);
    packed_mpf_r =  allocbuf_mpf(MPF_PREC, 2);

    mpf_t R,Sigma,b,tau,zero,one;
    mpf_t r[2],div,h1,h2,h3;
    mpf_t u1,u2,time,T,time1,T1,temp,Abs;
    mpf_init_set_str(r[0], "0.0",10);
    mpf_init_set_str(r[1], "0.0",10);
    mpf_init(div);
    mpf_init(h1);
    mpf_init(h2);
    mpf_init(h3);
    mpf_init(T);
    mpf_init(time1);
    mpf_init(T1);
    mpf_init(temp);
    mpf_init(Abs);

    mpf_init_set_str(b,"0.0",10);
    mpf_init_set_str(one,"1.0",10);
    mpf_init_set_str(zero,"0.0",10);
    mpf_init_set_str(time,"0.0",10);
    mpf_init_set_str(tau,"0.01",10);
    mpf_init_set_str(R,"28.0",10);
    mpf_init_set_str(Sigma,"10.0",10);
    mpf_init_set_str(u1,"8.0",10);
    mpf_init_set_str(u2,"3.0",10);
    mpf_div(u1,u1,u2);
    mpf_set(b,u1);
    mpf_clear(u1);
    mpf_clear(u2);

    mpf_t x[N+1],y[N+1],z[N+1];

    for (i = 0; i<N+1; i++)
    {
       mpf_init(x[i]);
       mpf_init(y[i]);
       mpf_init(z[i]);
    }

    mpf_t sum[pad*Numt],tempv[pad*Numt];

    #pragma omp parallel for schedule(static)
    for (i = 0; i<pad*Numt; i++)
    {
        mpf_init_set(tempv[i],zero);
    }

    #pragma omp parallel for schedule(static)
    for (i = 0; i<pad*Numt; i++)
    {
        mpf_init_set(sum[i],zero);
    }

    mpf_set_str(T1,"10.0",10);
    mpf_set_str(time1,"10.0",10);
    mpf_set_str(T,"200.001",10);
    mpf_set_str(x[0],"-15.8",10);
    mpf_set_str(y[0],"-17.48",10);
    mpf_set_str(z[0],"35.64",10);

    /////////////////////////////////////////////////////////////////////

    mpf_set_str(time,"0.0",10);
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = - MPI_Wtime();

    l=0;
    while (mpf_cmp(time,T)<0)
    {
         #pragma omp parallel private(i,j,tid,n,nt,shift,sh,ifinalt,istartt)
         {
         tid = omp_get_thread_num();
         nt =  omp_get_num_threads()/2;
         for (i = 0; i<N; i++)
         {
            #pragma omp single
            {
               istart=(rank*(i+1))/size;
               ifinal=((rank+1)*(i+1))/size -1;
            }
            ///////// SPMD pattern for threads too /////////////
            if(tid<nt)
            {
            istartt=istart+(tid*(ifinal-istart+1))/nt;
            ifinalt=istart+((tid+1)*(ifinal-istart+1))/nt -1;
            for (j=istartt; j<=ifinalt; j++)
            {
                mpf_mul(tempv[pad*tid],x[i-j],z[j]);
                mpf_add(sum[pad*tid],sum[pad*tid],tempv[pad*tid]);
            }
            }
            else
            {
            istartt=istart+((tid-nt)*(ifinal-istart+1))/nt;
            ifinalt=istart+((tid-nt+1)*(ifinal-istart+1))/nt -1;
            for (j=istartt; j<=ifinalt; j++)
            {
                mpf_mul(tempv[pad*tid],x[i-j],y[j]);
                mpf_add(sum[pad*tid],sum[pad*tid],tempv[pad*tid]);
            }
            }

            # pragma omp barrier
            //! Explicit Parallel Reduction for two sums for log(p) additions
            //! The first step is in a butterfly form
            sh=nt;
            n=nt;
            shift=(n+1)/2;
            while (n>1)
            {
                  if (tid <=n-1-shift)
                  {
                        mpf_add(sum[pad*tid],sum[pad*tid],sum[pad*(tid+shift)]);
                  }
                  else if (tid>=sh && tid<=sh+n-1-shift)
                  {
                        mpf_add(sum[pad*tid],sum[pad*tid],sum[pad*(tid+shift)]);
                  }
                  n=shift;
                  shift=(n+1)/2;
                  # pragma omp barrier
            }
            /// End of explicit Parallel Reduction for two sums for log(p) additions

            /// Overlapped MPI_All_reduce with preparatory work for x[i+1],y[i+1],z[i+1]
            /// Work with at least 4 threads
            if(tid==0)
            {
            mpf_set(sum[1],sum[pad*nt]);

            //!MPI_ALLREDUCE for sum[0] and sum[1]
            pack_mpf(sum[0], 2, packed_mpf_s);
            MPI_Allreduce(packed_mpf_s, packed_mpf_r, 2, MPI_MPF, MPI_MPF_SUM, MPI_COMM_WORLD);
            unpack_mpf(packed_mpf_r, sum[0], 2);
            //!END MPI_ALLREDUCE

            }
            else if(tid==1)
            {
                mpf_sub(h1,y[i],x[i]);
                mpf_div_ui(h1,h1,i+1);
                mpf_mul(x[i+1],h1,Sigma);

            }
            else if(tid==2)
            {
                 mpf_mul(h2,R,x[i]);
                 mpf_sub(h2,h2,y[i]);

            }
            else if(tid==3)
            {
                 mpf_mul(h3,b,z[i]);
            }

            #pragma omp barrier

            if (tid==2)
            {
                mpf_sub(h2,h2,sum[0]);
                mpf_div_ui(y[i+1],h2,i+1);
            }
            else if(tid==3)
            {
                mpf_sub(h3,sum[1],h3);
                mpf_div_ui(z[i+1],h3,i+1);
            }
            #pragma omp barrier

            mpf_set(sum[pad*tid],zero);

        }

        //! determining the step size and weather to print;
        #pragma omp single
        {
           //-------------------------------------------
           mpf_abs (Abs, x[N-1]);
           mpf_set(temp,Abs);

           mpf_abs (Abs, y[N-1]);
           if(mpf_cmp (Abs, temp)>0) mpf_set(temp,Abs);

           mpf_abs (Abs, z[N-1]);
           if(mpf_cmp (Abs, temp)>0) mpf_set(temp,Abs);

           mpf_get_str (ch, &Exp, 2, 10, temp);
           mpf_div_2exp (temp, temp, Exp);
           d=1.0/mpf_get_d (temp);
           t1=pow(d,1.0/((double)(N-1)));
           t1=t1*pow(2.0,-Exp/(double)(N-1));
           //-------------------------------------------
           mpf_abs (Abs, x[N]);
           mpf_set(temp,Abs);

           mpf_abs (Abs, y[N]);
           if(mpf_cmp (Abs, temp)>0) mpf_set(temp,Abs);

           mpf_abs (Abs, z[N]);
           if(mpf_cmp (Abs, temp)>0) mpf_set(temp,Abs);

           mpf_get_str (ch, &Exp, 2, 10, temp);
           mpf_div_2exp (temp, temp, Exp);
           d=1.0/mpf_get_d (temp);
           t2=pow(d,1.0/((double)(N)));
           t2=t2*pow(2.0,-Exp/(double)(N));
           //-------------------------------------------
           if(t2<t1) t1=t2;
           t1=t1/exp(2.0);
           t1=t1*0.993;
           mpf_set_d (tau, t1);

           mpf_add(temp,time,tau);
           if(mpf_cmp (temp, time1)>=0)
           {
               mpf_sub(tau, time1, time);
               mpf_add(time1, time1, T1);
               l=1;
           }
         }
        //! end of determining the step size and weather to print;
        /////////////////////////////////////////////////
        //! One step forward with Horner's rule
        #pragma omp sections
        {
        #pragma omp section
        {
             mpf_set(h1,x[N]);
             for (j=N-1; j>=0; j--)
             {
                  mpf_mul(h1,h1,tau);
                  mpf_add(h1,h1,x[j]);
             }
             mpf_set(x[0],h1);
        }
        #pragma omp section
        {
              mpf_set(h2,y[N]);
              for (j=N-1; j>=0; j--)
              {
                    mpf_mul(h2,h2,tau);
                    mpf_add(h2,h2,y[j]);
              }
              mpf_set(y[0],h2);
        }
        #pragma omp section
        {
               mpf_set(h3,z[N]);
               for (j=N-1; j>=0; j--)
               {
                     mpf_mul(h3,h3,tau);
                     mpf_add(h3,h3,z[j]);
               }
               mpf_set(z[0],h3);
        }

        }
        }

        mpf_add(time,time,tau);
        if(l==1)
        {
          if(rank==0)
          {
             mpf_out_str(outfile,10,10,time);
             fprintf(outfile,"\n");
             mpf_out_str(outfile,10,50,x[0]);
             fprintf(outfile,"\n");
             mpf_out_str(outfile,10,50,y[0]);
             fprintf(outfile,"\n");
             mpf_out_str(outfile,10,50,z[0]);
             fprintf(outfile,"\n");
          }
          l=0;
        }

    }


    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time += MPI_Wtime();
    if (rank==0) printf("Time = %f.\n",elapsed_time);

    mpf_clear(r[0]);
    mpf_clear(r[1]);
    mpf_clear(div);
    mpf_clear(h1);
    mpf_clear(h2);
    mpf_clear(h3);
    mpf_clear(T);
    mpf_clear(one);
    mpf_clear(zero);
    mpf_clear(time);
    mpf_clear(tau);
    mpf_clear(R);
    mpf_clear(Sigma);
    mpf_clear(b);
    mpf_clear(time1);
    mpf_clear(T1);
    mpf_clear(temp);
    mpf_clear(Abs);

    for (i = 0; i<N+1; i++)
    {
        mpf_clear(x[i]);
        mpf_clear(y[i]);
        mpf_clear(z[i]);
    }

    for (i = 0; i<pad*Numt; i++)
    {
        mpf_clear(tempv[i]);
    }

    for (i = 0; i<pad*Numt; i++)
    {
        mpf_clear(sum[i]);
    }

    free_mpf(&(MPI_MPF));
    free_mpf_op(&(MPI_MPF_SUM));
    free(packed_mpf_s);
    free(packed_mpf_r);

    MPI_Finalize();

    fclose(outfile);


    return 0;
}






