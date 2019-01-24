#include "mpi.h"
#include <stdio.h>
#include <math.h>
#define mmax 20 //The maximum size of the matrix
#define pmax 8 //The maximum number of the processes
#define epsilon 1e-5

double inner_product(int, double [], double []);
void matrix_vector_product(int, int, double [][mmax], double[], double[]);
void input(int *, double [][mmax], double[]);
void output(int, double[]);
void process(int, double[][mmax], double[], double[], int, int);

int main(int argc, char *argv[])
{
//Initialize
  int i,j,k,n,myid,np,current;
  double matrix[mmax][mmax],vector[mmax],solution[mmax];
  double d1,d2,n1,n2,s;
  int disp[pmax],counts[pmax];
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&np);


  if (0==myid) input(&n, matrix, vector); //Input data from external file
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  process(n, matrix, vector, solution, np, myid); //Process
  if (0==myid) output(n, solution); //Output

  MPI_Finalize();
  return 0;  
}

void process(int n, double matrix[][mmax], double vector[], double solution[], int np, int myid) 
{
  int disp[pmax], counts[pmax];
  int m,i,j;
  double A[mmax][mmax], myvector[mmax], g[mmax], gx[mmax], x[mmax], gd[mmax], d[mmax], t[mmax];
  double n1, n2, d1, d2, product0, s;
  MPI_Datatype vectype;

//Define derived datatype : a vector as a row of double data
  MPI_Type_contiguous(mmax,MPI_DOUBLE,&vectype);
  MPI_Type_commit(&vectype);

//fill in values for array disp[] and counts[] which are to be used in
//MPI_Scatterv and MPI_AllgatherV
  for (i=0; i!=np; i++)
  {
     disp[i]=n/np*i;
     counts[i]=(i!=np-1)?(n/np):(n/np+n%np);
  }
  m = counts[myid];
  
  MPI_Scatterv(matrix, counts, disp, vectype, A, counts[myid], vectype, 0, MPI_COMM_WORLD);
  MPI_Scatterv(vector, counts, disp, MPI_DOUBLE, myvector, counts[myid], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for (i=0; i!=m; i++) {
    d[i]=0; x[i]=0; t[i]=0;
  }

  for (i=0; i!=m; i++) g[i]=-myvector[i];

  for (j=1; j!=n+1; j++) {
    product0=inner_product(m, g, g); d1=0;
    MPI_Allreduce(&product0, &d1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   
  //In a matrix-vector product, the vector must be a full one instead of seperate ones,
  //and as a result MPI_Allgatherv is used.
    MPI_Allgatherv(x, m, MPI_DOUBLE, gx, counts, disp, MPI_DOUBLE, MPI_COMM_WORLD);
    matrix_vector_product(m, n, A, gx, g);

    for (i=0; i!=m; i++) g[i]=g[i]-myvector[i];
    product0=inner_product(m, g, g); n1=0;
    MPI_Allreduce(&product0, &n1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


    for (i=0; i!=m; i++) d[i]=-g[i]+n1/d1*d[i];
    product0=inner_product(m, d, g); n2=0;
    MPI_Allreduce(&product0, &n2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allgatherv(d, m, MPI_DOUBLE, gd, counts, disp, MPI_DOUBLE, MPI_COMM_WORLD);
    matrix_vector_product(m, n, A, gd, t);
    
    product0=inner_product(m, d, t); d2=0;
    MPI_Allreduce(&product0, &d2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    s=-n2/d2;

    for (i=0; i!=m; i++) x[i]=x[i]+s*d[i];
  }
  MPI_Gatherv(x, m, MPI_DOUBLE, solution, counts, disp, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return;
}

//Input
void input(int *n, double matrix[][mmax], double vector[]) {
  int i,j;
  FILE *fp;
  fp=fopen("cg.in","r");
  if (fp==NULL) {fclose(fp); return;}
  fscanf(fp,"%d",n);
  for (i=0; i!=*n; i++)
    for (j=0; j!=*n; j++)
      fscanf(fp,"%lf",&matrix[i][j]);
  for (i=0; i!=*n; i++) fscanf(fp,"%lf",&vector[i]);
  fclose(fp);
  return;
}

//Output
void output(int n, double solution[])
{
  int i;
  printf("x=");
  for (i=0; i!=n; i++) printf("%lf ", solution[i]);
  printf("\n");
  return; 
}

//Inner product
double inner_product(int m, double g[], double h[])
{
  double product=0;
  int i;
  for (i=0; i!=m; i++) product=product+g[i]*h[i];
  return product;
}

//Matrix-vector product
void matrix_vector_product(int m, int n, double A[][mmax], double B[], double C[])
{
  int i,j;
  for (i=0; i!=m; i++)
      C[i]=0;
  for (i=0; i!=m; i++)
    for (j=0; j!=n; j++) 
      C[i]=C[i]+A[i][j]*B[j];
  return;
}
